import ast
import numpy as np

import torch
from torch import optim, nn

import dgl

import pytorch_lightning as pl
import nni

from dgld.modules.networks.gcn import *
from dgld.modules.networks.gatv2 import *
from dgld.modules.networks.sage import *
from dgld.modules.networks.jknet import *
from dgld.modules.networks.mlp import *
from dgld.utils.evaluation import *
from dgld.data.dataloader import *
from dgld.modules.dglAug.augs import *


class GGD(pl.LightningModule):
    """自监督预训练方法Graph Group Discrimination
    
    对于下游任务，只需要取出其中的编码器（encoder）用于推断表征（embedding）即可。
    """
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
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore='graph')
        # data & dataloader
        self._prepare_dataloader(graph)
        # backbone
        if encoder_name == 'gcn':
            if n_layers > 1:
                self.encoder = GCN(
                    graph.ndata['feat'].shape[-1], 
                    embedding_dim*2, 
                    embedding_dim, 
                    dropout=dropout, 
                    num_layers=n_layers, 
                    batch_norm=batch_norm, 
                    act_fn=nn.PReLU()
                )
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
        elif encoder_name == 'gatv2':
            self.encoder = GATv2(n_layers, graph.ndata['feat'].shape[-1], 2*embedding_dim, embedding_dim, heads=[2]*n_layers, feat_drop=dropout, attn_drop=0)
        elif encoder_name == 'jknet':
            self.encoder = JKNet(
                graph.ndata['feat'].shape[-1], 
                embedding_dim*2, 
                embedding_dim, 
                dropout=dropout, 
                num_layers=n_layers, 
                mode='cat'
            )
        self.proj = MLP(embedding_dim, 2*embedding_dim, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
      
    def _prepare_dataloader(self, graph):
        self.graph = graph
        # self.label = self.graph.ndata.pop('label') # 避免label在batch中被shuffle或修改了而没察觉
        self.label = self.graph.ndata['label']
        
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=self.hparams.n_layers)
        self.train_transform = ContrastiveTransform([
            Compose([RemoveSelfLoop(), AddSelfLoop()]), # 原图或不丢失原始语义的数据增强
            Compose([NodeShuffle(), RemoveSelfLoop(), AddSelfLoop()]) # corruption / out-of-distribution
            ])
    
    def forward(self, batch, mode='train'):
        # compute embeddings
        if isinstance(batch, dgl.DGLGraph):
            embed_list = [self.encoder(batch, batch.ndata['feat'])]
        elif isinstance(batch[0], dgl.DGLGraph):
            embed_list = [self.encoder(g, g.ndata['feat']) for g in batch]
        elif isinstance(batch[0], list):
            embed_list = [self.encoder(view[-1], view[-1][0].srcdata['feat']) for view in batch]
                
        sc_list = [self.proj(embed).reshape(1,-1) for embed in embed_list] # proj
        
        return sc_list
        
    def training_step(self, batch, batch_idx):
        sc_list = self(batch, mode='train')
        # compute loss
        lbl_list = [torch.ones_like(sc_list[0]), torch.zeros_like(sc_list[1])]
        
        logits = torch.cat(sc_list, dim=1)
        labels = torch.cat(lbl_list, dim=1)
        
        loss = self.loss_fn(logits, labels) 
        self.log('train_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    # def configure_optimizers(self):
    #     optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    #     lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
    #     )
    #     return [optimizer], [lr_scheduler]
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitGGD")
        # model specific args
        parser.add_argument("--encoder_name", type=str, default='gcn')
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--batch_norm", type=ast.literal_eval, default=False)
        # training specific args
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--batch_size", type=int, default=0)
        return parent_parser
            
    def train_dataloader(self):
        train_dataloader = MultiViewDataloader(self.graph, self.train_transform, batch_size=self.hparams.batch_size, sampler=self.sampler, shuffle=True, align=False)
        return train_dataloader