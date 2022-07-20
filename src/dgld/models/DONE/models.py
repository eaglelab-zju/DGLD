import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import dgl
import dgl.function as fn

# TODO: add .
from done_utils import random_walk_with_restart, train_step, test_step

# TODO: del
from sklearn.metrics import roc_auc_score
import numpy as np
from done_utils import load_paper_dataset, recall_at_k
import warnings
warnings.filterwarnings("ignore")

class DONE():
    def __init__(self, 
                 feat_size:int,
                 num_nodes:int,
                 embedding_dim=32,
                 num_layers=2,
                 activation=nn.LeakyReLU(negative_slope=0.2),
                 ):
        self.model = DONE_Base(feat_size, num_nodes, embedding_dim, num_layers, activation)
    
    def fit(self, 
            graph:dgl.DGLGraph,
            lr=1e-3,
            weight_dacay=0,
            num_epoch=1,
            num_neighbors=2,
            alpha1=0.2,
            alpha2=0.2,
            alpha3=0.2,
            alpha4=0.2,
            alpha5=0.2,
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
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_dacay)
        
        writer = SummaryWriter(log_dir=logdir)
        
        # newg = random_walk_with_restart(graph, eps=0)
        newg = graph
        
        for epoch in range(num_epoch):
            score, loss = train_step(self.model, optimizer, graph, batch_size, alphas=[alpha1, alpha2, alpha3, alpha4, alpha5], newg=newg)
            
            if y_true is not None:
                auc = roc_auc_score(y_true, score)
                print(f"Epoch: {epoch:04d}, train/loss={loss:.5f}, train/auc: {auc:.5f}")
                
            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/auc', auc, epoch)
    
    def predict(self, 
                graph:dgl.DGLGraph, 
                batch_size:int, 
                alpha1=0.2,
                alpha2=0.2,
                alpha3=0.2,
                alpha4=0.2,
                alpha5=0.2,
                ):
        print('*'*20,'predict','*'*20)
        
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!') 
            
        predict_score = test_step(self.model, graph, batch_size, alphas=[alpha1, alpha2, alpha3, alpha4, alpha5])
        
        return predict_score
   
    
class DONE_Base(nn.Module):
    def __init__(self, feat_size, num_nodes, hid_feats, num_layers, activation):
        super(DONE_Base, self).__init__()
        
        self.attr_encoder = self._add_mlp(feat_size, hid_feats, hid_feats, num_layers, activation)
        
        self.attr_decoder = self._add_mlp(hid_feats, hid_feats, feat_size, num_layers, activation)
        
        self.struct_encoder = self._add_mlp(num_nodes, hid_feats, hid_feats, num_layers, activation)
        
        self.struct_decoder = self._add_mlp(hid_feats, hid_feats, num_nodes, num_layers, activation)
    
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
        # structure
        h_s = self.struct_encoder(x)
        x_hat = self.struct_decoder(h_s)
        g.ndata['h'] = h_s
        g.update_all(self._homophily_loss_message_func, fn.mean('hh', 'h_str'))
        h_str = g.ndata.pop('h_str')
        # attribute
        h_a = self.attr_encoder(c)
        c_hat = self.attr_decoder(h_a)
        g.ndata['h'] = h_a
        g.update_all(self._homophily_loss_message_func, fn.mean('hh', 'h_attr'))
        h_attr = g.ndata.pop('h_attr')
        return h_s, x_hat, h_a, c_hat, h_str, h_attr
    
    def _add_mlp(self, in_feats, hid_feats, out_feats, num_layers, activation):
        assert(num_layers >= 2)
        mlp = nn.Sequential()
        # input layer
        mlp.add_module('linear_in', nn.Linear(in_feats, hid_feats))
        mlp.add_module('act_in', activation)
        # hidden layers
        for i in range(num_layers-2):
            mlp.add_module(f'linear_hid_{i}', nn.Linear(hid_feats, hid_feats))
            mlp.add_module(f'act_hid_{i}', activation)
        # output layer
        mlp.add_module('linear_out', nn.Linear(hid_feats, out_feats))
        mlp.add_module('act_out', activation)
        return mlp
    
    def _homophily_loss_message_func(self, edges):
        return {'hh': torch.norm(edges.src['h'] - edges.dst['h'], dim=1)}


g, indices = load_paper_dataset('cora')
# newg = random_walk_with_restart(g)
feat = g.ndata['feat']
label = g.ndata.pop('label')
label_aug = np.zeros_like(label)
indices = indices.squeeze()
# print(indices)
label_aug[indices > 2708] = 1
# adj = newg.adj().to_dense()
num_nodes = g.number_of_nodes()
# model = DONE_Base(feat.shape[1], num_nodes, 32, 2, nn.LeakyReLU(negative_slope=0.2))
# print(model)
# h_a, x_hat, h_s, adj_hat, h_str, h_attr = model(g, feat, adj)
# print(h_a.shape, x_hat.shape, h_s.shape, adj_hat.shape, h_str.shape, h_attr.shape)
model = DONE(feat.shape[1], num_nodes)
model.fit(g, batch_size=num_nodes, num_epoch=200, lr=0.03, y_true=label_aug)
score = model.predict(g, batch_size=num_nodes)
auc = roc_auc_score(label_aug, score)
print(f"auc: {auc:.5f}")
for p in [5, 10, 15, 20, 25]:
    k = int(num_nodes * p / 100.0)
    recall = recall_at_k(label_aug, score, k)
    print(f"Recall@{p}%: {recall:.5f}")