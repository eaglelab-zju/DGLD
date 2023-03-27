import ast
import numpy as np

import torch
from torch import optim, nn
import dgl
import pytorch_lightning as pl
import nni

from dgld.modules.networks.gcn import *
from dgld.modules.networks.sage import *
from dgld.modules.networks.jknet import *
from dgld.modules.networks.gatv2 import *
from dgld.modules.networks.mlp import *
from dgld.utils.evaluation import *
from dgld.data.dataloader import *
from dgld.modules.dglAug.augs import *

class GRACE(pl.LightningModule):
    def __init__(self,
                 graph,
                 # model parameters
                 embedding_dim,
                 n_layers,
                 dropout,
                 batch_norm,
                 encoder_name,
                 # training parameters
                 lr,
                 weight_decay,
                 batch_size,
                 inst_tau,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore='graph')
        self._prepare(graph)
        
        # encoder
        if encoder_name == 'gcn':
            if n_layers > 1:
                self.encoder = GCN(graph.ndata['feat'].shape[-1], embedding_dim*2, embedding_dim, dropout=dropout, num_layers=n_layers, batch_norm=batch_norm)
            else:
                self.encoder = GraphConv(graph.ndata['feat'].shape[-1], embedding_dim, activation=nn.ReLU())
        elif encoder_name == 'sage':
            if n_layers > 1:
                self.encoder = SAGE(
                    graph.ndata['feat'].shape[-1], 
                    embedding_dim*2, 
                    embedding_dim, 
                    dropout=dropout, 
                    num_layers=n_layers, 
                    batch_norm=batch_norm, 
                    act_fn=nn.PReLU()
                ) 
            else: 
                self.encoder = SAGEConv(graph.ndata['feat'].shape[-1], embedding_dim, 'mean', activation=nn.ReLU())
        elif encoder_name == 'jknet':
            self.encoder = JKNet(
                graph.ndata['feat'].shape[-1], 
                embedding_dim*2, 
                embedding_dim, 
                dropout=dropout, 
                num_layers=n_layers, 
                mode='cat'
            )
        # projector
        self.proj = MLP(embedding_dim, embedding_dim*2, embedding_dim, dropout, batch_norm, num_layers=2) 
    
    def _prepare(self, graph):
        self.graph = graph
        # self.label = self.graph.ndata.pop('label') # 避免label在batch中被shuffle或修改了而没察觉
        self.label = self.graph.ndata['label']
        # self.hparams.cluster_num = list(map(int, self.hparams.cluster_num.split(',')))
        if self.hparams.batch_size==0 or self.hparams.batch_size>=self.graph.num_nodes():
            self.hparams.batch_size = self.graph.num_nodes()
        # self.mask = self.mask_correlated_samples(self.hparams.batch_size)
        # if self.graph.num_nodes()%self.hparams.batch_size!=0:
        #     self.mask_last = self.mask_correlated_samples(self.graph.num_nodes()%self.hparams.batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=self.hparams.n_layers)
        self.train_transform = ContrastiveTransform([
            Compose([FeatMask(0.2), DropEdge(0.2), RemoveSelfLoop(), AddSelfLoop()]), # 原图
            Compose([FeatMask(0.2), DropEdge(0.2), RemoveSelfLoop(), AddSelfLoop()]) # 不丢失原始语义的数据增强
            ])
        if hasattr(self.hparams, 'train_transform'):
            self.train_transform = self.hparams.train_transform
      
    def forward(self, batch, mode='all'):
        # encoder: compute embeddings
        if isinstance(batch, dgl.DGLGraph):
            embed_list = [self.encoder(batch, batch.ndata['feat'])]
        elif isinstance(batch[0], dgl.DGLGraph):
            embed_list = [self.encoder(g, g.ndata['feat']) for g in batch]
        elif isinstance(batch[0], list):
            embed_list = [self.encoder(view[-1], view[-1][0].srcdata['feat']) for view in batch]
        
        # projectors: outputs
        outputs = {}                  
                
        for i, embed in enumerate(embed_list): # for each embed in different views
            key = f"view{i+1}_feat"
            outputs[key] = self.proj(embed)
            
        return outputs
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask
    
    def info_nce(self, z1, z2, mode='train'):
        """instance discrimination loss

        Parameters
        ----------
        z1 : torch.Tensor
            view1's embedding vector, of shape (B, D)
        z2 : torch.Tensor
            view2's embedding vector, of shape (B, D)

        Returns
        -------
        losses : torch.Tensor
            no reduction loss, of shape (B,)
        """
        # compute logits
        N = 2 * z1.shape[0]
        z = torch.cat((z1, z2), dim=0)

        sim = torch.matmul(z, z.T) / self.hparams.inst_tau
        sim_i_j = torch.diag(sim, z1.shape[0])
        sim_j_i = torch.diag(sim, -z1.shape[0])

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(z1.shape[0])
        # mask = self.mask if z1.shape[0]==self.hparams.batch_size else self.mask_last
        negative_samples = sim[mask].reshape(N, -1)

        # a tensor of all zeros, of shape (N,)
        labels = torch.zeros(N).to(self.device).long() 
        # similarity matrix of anchor and positive (column 0) and negative (column 1~N-2) samples, of shape (N, N-1)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        losses = self.criterion(logits, labels).reshape(z1.shape[0], -1)
        losses = torch.mean(losses, dim=1) 
        
        return losses
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        # compute loss
        # InfoNCE
        losses = self.info_nce(z1=outputs['view1_feat'], z2=outputs['view2_feat'])
        # if self.hparams.weight is not None:
        #     losses *= self.hparams.weight
        loss = losses.mean()
        self.log('loss/training loss', loss, reduce_fx='sum')
        return loss
      
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GRACE")
        # model specific args
        parser.add_argument("--encoder_name", type=str, default='gcn')
        parser.add_argument("--embedding_dim", type=int, default=128)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--batch_norm", type=ast.literal_eval, default=False)
        # training specific args
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--batch_size", type=int, default=8192)
        parser.add_argument("--inst_tau", type=float, default=0.5)
        return parent_parser
        
    def train_dataloader(self):
        train_dataloader = MultiViewDataloader(self.graph, self.train_transform, batch_size=self.hparams.batch_size, sampler=self.sampler, shuffle=True, align=True, drop_last=True)
        return train_dataloader