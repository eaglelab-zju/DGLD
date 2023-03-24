import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv

# Multi-layer Graph Convolutional Networks
class GCN(nn.Module):
    def __init__(self, 
                 in_dim, 
                 hid_dim,
                 out_dim, 
                 num_layers,
                 dropout,
                 batch_norm,
                 act_fn=F.relu):
        super(GCN, self).__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim))
        
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(hid_dim))
        
        for _ in range(self.num_layers - 2):
            self.convs.append(GraphConv(hid_dim, hid_dim))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(hid_dim))

        self.convs.append(GraphConv(hid_dim, out_dim))
        
        self.act_fn = act_fn
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, h):
        for i in range(self.num_layers):
            if isinstance(g, list):
                h = self.convs[i](g[i], h)
            else:
                h = self.convs[i](g, h)
                
            if i < self.num_layers - 1 and self.batch_norm:
                h = self.bns[i](h)    
                
            h = self.act_fn(h)
            
            if i < self.num_layers - 1:
                h = self.dropout(h)
            
        return h