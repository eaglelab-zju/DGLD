import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import dgl
import dgl.function as fn

# TODO: add .
from .done_utils import random_walk_with_restart, train_step, test_step

# TODO: del
from sklearn.metrics import roc_auc_score
import numpy as np
def recall_at_k(truth, score, k):
    ranking = np.argsort(-score)
    top_k = ranking[:k]
    top_k_label = truth[top_k]
    return top_k_label.sum() / truth.sum()

class DONE():
    def __init__(self, 
                 feat_size:int,
                 num_nodes:int,
                 embedding_dim=32,
                 num_layers=2,
                 activation=nn.LeakyReLU(negative_slope=0.2),
                 dropout=0.,
                 ):
        self.model = DONE_Base(feat_size, num_nodes, embedding_dim, num_layers, activation, dropout)
    
    def fit(self, 
            graph:dgl.DGLGraph,
            lr=1e-3,
            weight_decay=0.,
            num_epoch=1,
            num_neighbors=-1,
            alphas=[0.2]*5,
            logdir='tmp',
            batch_size=0,
            device='cpu',
            y_true=None,
            ):
        print('*'*20,'training','*'*20)
        
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')
        
        if batch_size == 0:
            batch_size = graph.number_of_nodes()
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        writer = SummaryWriter(log_dir=logdir)
        
        # adj = random_walk_with_restart(graph)
        # print(adj.shape)
        adj = graph.adj().to_dense()
        
        # pretrain w/o the outlier scores
        for epoch in range(num_epoch):
            score, loss = train_step(self.model, optimizer, graph, adj, batch_size, alphas, num_neighbors, device, pretrain=True)
            print(f"Epoch: {epoch:04d}, pretrain/loss={loss:.5f}")
        
        # train with the outlier scores
        for epoch in range(num_epoch):
            score, loss = train_step(self.model, optimizer, graph, adj, batch_size, alphas, num_neighbors, device)
            if y_true is not None and len(y_true.shape)==1:
                self.model.eval()
                recall = recall_at_k(y_true, score, int(adj.shape[0] * 0.05))
                auc = roc_auc_score(y_true, score)
                print(f"Epoch: {epoch:04d}, train/loss={loss:.5f}, train/auc: {auc:.5f}, train/recall@5%={recall:.5f}")
                writer.add_scalar('train/auc', auc, epoch)
                writer.add_scalar('train/recall', recall, epoch)
            else:
                print(f"Epoch: {epoch:04d}, train/loss={loss:.5f}")
                
            writer.add_scalar('train/loss', loss, epoch)
            writer.flush()
    
    def predict(self, 
                graph:dgl.DGLGraph, 
                batch_size:int, 
                device='cpu',
                alphas=[0.2]*5,
                ):
        print('*'*20,'predict','*'*20)
        
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!') 
            
        if batch_size == 0:
            batch_size = graph.number_of_nodes()
        
        # adj = random_walk_with_restart(graph)
        adj = graph.adj().to_dense()
            
        predict_score = test_step(self.model, graph, adj, batch_size, alphas, device)
        
        return predict_score
   
    
class DONE_Base(nn.Module):
    def __init__(self, feat_size, num_nodes, hid_feats, num_layers, activation, dropout):
        super(DONE_Base, self).__init__()
        
        self.attr_encoder = self._add_mlp(feat_size, hid_feats, hid_feats, num_layers, activation, dropout)
        
        self.attr_decoder = self._add_mlp(hid_feats, hid_feats, feat_size, num_layers, activation, dropout)
        
        self.struct_encoder = self._add_mlp(num_nodes, hid_feats, hid_feats, num_layers, activation, dropout)
        
        self.struct_decoder = self._add_mlp(hid_feats, hid_feats, num_nodes, num_layers, activation, dropout)
    
    def forward(self, g, x, c):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph data
        x : torch.Tensor
            adjacency matrix
        c : torch.Tensor
            attribute matrix

        """
        with g.local_scope():
            # structure
            h_s = self.struct_encoder(x)
            x_hat = self.struct_decoder(h_s)
            g.ndata['h'] = h_s
            g.update_all(self._homophily_loss_message_func, fn.mean('hh', 'h_str'))
            
            # attribute
            h_a = self.attr_encoder(c)
            c_hat = self.attr_decoder(h_a)
            g.ndata['h'] = h_a
            g.update_all(self._homophily_loss_message_func, fn.mean('hh', 'h_attr'))
            
            return h_s, x_hat, h_a, c_hat, g.ndata['h_str'], g.ndata['h_attr']
    
    def _add_mlp(self, in_feats, hid_feats, out_feats, num_layers, activation, dropout):
        assert(num_layers >= 2)
        mlp = nn.Sequential()
        # input layer
        mlp.add_module('linear_in', nn.Linear(in_feats, hid_feats))
        mlp.add_module('act_in', activation)
        # hidden layers
        for i in range(num_layers-2):
            mlp.add_module(f'dropout_hid_{i}', nn.Dropout(dropout))
            mlp.add_module(f'linear_hid_{i}', nn.Linear(hid_feats, hid_feats))
            mlp.add_module(f'act_hid_{i}', activation)
        # output layer
        mlp.add_module('dropout_out', nn.Dropout(dropout))
        mlp.add_module('linear_out', nn.Linear(hid_feats, out_feats))
        mlp.add_module('act_out', activation)
        return mlp
    
    def _homophily_loss_message_func(self, edges):
        return {'hh': torch.norm(edges.src['h'] - edges.dst['h'], dim=1)}
