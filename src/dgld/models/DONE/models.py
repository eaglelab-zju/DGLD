from tabnanny import verbose
import torch
import torch.nn as nn

import dgl
import dgl.function as fn

from .done_utils import random_walk_with_restart, train_step, test_step

from dgld.utils.early_stopping import EarlyStopping

class DONE():
    def __init__(self, 
                 feat_size:int,
                 num_nodes:int,
                 embedding_dim=32,
                 num_layers=2,
                 activation=nn.LeakyReLU(negative_slope=0.2),
                 dropout=0.,
                 ):
        """Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding

        Parameters
        ----------
        feat_size : int
            dimension of feature
        num_nodes : int
            number of nodes
        embedding_dim : int, optional
            dimension of embedding, by default 32
        num_layers : int, optional
            number of layers of the auto-encoder, where the number of layers of the encoder and decoder is the same number, by default 2
        activation : torch.nn.quantized.functional, optional
            activation function, by default nn.LeakyReLU(negative_slope=0.2)
        dropout : float, optional
            rate of dropout, by default 0.
        """
        self.model = DONE_Base(feat_size, num_nodes, embedding_dim, num_layers, activation, dropout)
    
    def fit(self, 
            graph:dgl.DGLGraph,
            lr=1e-3,
            weight_decay=0.,
            num_epoch=1,
            num_neighbors=-1,
            alphas=[0.2]*5,
            batch_size=0,
            max_len=0, 
            restart=0.5,
            device='cpu',
            ):
        """Fitting model

        Parameters
        ----------
        graph : dgl.DGLGraph
            graph data
        lr : float, optional
            learning rate, by default 1e-3
        weight_decay : float, optional
            weight decay (L2 penalty), by default 0.
        num_epoch : int, optional
            number of training epochs, by default 1
        num_neighbors : int, optional
            number of sampling neighbors, by default -1
        alphas : list, optional
            balance parameters, by default [0.2]*5
        batch_size : int, optional
            the size of training batch, by default 0
        max_len : int, optional
            the maximum length of the truncated random walk, if the value is zero, the adjacency matrix of the original graph is used, by default 0
        restart : float, optional
            probability of restart, by default 0.5
        device : str, optional
            device of computation, by default 'cpu'
        """
        print('*'*20,'training','*'*20)
        
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')
            
        self.model.to(device)
        
        if batch_size == 0:
            batch_size = graph.number_of_nodes()
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        
        # preprocessing
        # graph = graph.remove_self_loop().add_self_loop()
        if max_len > 0:
            adj = random_walk_with_restart(graph, k=max_len, r=1-restart)
        else:
            adj = graph.adj().to_dense()

        early_stop = EarlyStopping(early_stopping_rounds=10, patience=100)

        # pretrain w/o the outlier scores
        for epoch in range(num_epoch):
            score, loss = train_step(self.model, optimizer, graph, adj, batch_size, alphas, num_neighbors, device, pretrain=True)
            print(f"Epoch: {epoch:04d}, pretrain/loss={loss:.5f}")
        
        # train with the outlier scores
        for epoch in range(num_epoch):
            score, loss = train_step(self.model, optimizer, graph, adj, batch_size, alphas, num_neighbors, device)
            print(f"Epoch: {epoch:04d}, train/loss={loss:.5f}")
            
            early_stop(loss, self.model)
            if early_stop.isEarlyStopping():
                print(f"Early stopping in round {epoch}")
                break
            
    
    def predict(self, 
                graph:dgl.DGLGraph, 
                batch_size=0, 
                max_len=0, 
                restart=0.5,
                device='cpu',
                alphas=[0.2]*5,
                ):
        """predict and return anomaly score of each node

        Parameters
        ----------
        graph : dgl.DGLGraph
            graph data
        batch_size : int, optional
            the size of training batch, by default 0
        max_len : int, optional
            the maximum length of the truncated random walk, if the value is zero, the adjacency matrix of the original graph is used, by default 0
        restart : float, optional
            probability of restart, by default 0.5
        device : str, optional
            device of computation, by default 'cpu'
        alphas : list, optional
            balance parameters, by default [0.2]*5

        Returns
        -------
        predict_score : numpy.ndarray
            predicted outlier score
        """
        print('*'*20,'predict','*'*20)
        
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!') 
            
        self.model.to(device)
            
        if batch_size == 0:
            batch_size = graph.number_of_nodes()
        
        if max_len > 0:
            adj = random_walk_with_restart(graph, k=max_len, r=1-restart)
        else:
            adj = graph.adj().to_dense()
            
        predict_score = test_step(self.model, graph, adj, batch_size, alphas, device)
        
        return predict_score
   
    
class DONE_Base(nn.Module):
    """This is a basic structure model of DONE.

    Parameters
    ----------
    feat_size : int
        the feature dimension of the input data
    num_nodes : int
        number of nodes
    hid_feats : int
        the feature dimension of the hidden layers
    num_layers : int
        number of layers of the auto-encoder, where the number of layers of the encoder and decoder is the same number
    activation : torch.nn.quantized.functional
        activation function
    dropout : float
        probability of restart
    """
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
            structure matrix
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
