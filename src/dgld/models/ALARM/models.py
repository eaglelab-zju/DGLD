"""
A Deep Multi-View Framework for Anomaly Detection on Attributed Networks.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
# from layers import GraphConvolution
from dgl.nn.pytorch import GraphConv
from .alarm_utils import train_step, test_step, normalize_adj
from utils.early_stopping import EarlyStopping

class Encoder(nn.Module):
    """Encoder of DominantModel

    Parameters
    ----------
    nfeat : int
        dimension of feature 
    nhid : int
        dimension of hidden embedding
    dropout : float
        Dropout rate
    view_num : int
        number of view, by default 3
    agg_type: int
        aggregator type, by default 0. 0: Concatention | 1: Random aggregation weights | 2: Manual aggregation weights
    agg_vec : list
        weighted aggregation vector, bydefault: [1,1,1], stands for concatention. This is necessary if agg_type is 2.
    """
    def __init__(self, nfeat, nhid, dropout, view_num, agg_type, agg_vec):
        super(Encoder, self).__init__()
        
        self.view_num = view_num
        self.agg_type = agg_type
        self.nhid = nhid
        self.agg_vec = agg_vec
        
        self.single_view = int(nfeat / view_num)
        self.view_feat = [self.single_view for i in range(view_num - 1)]
        self.view_feat.append(int(nfeat / view_num) + int(nfeat % view_num))
        
        self.gc1 = nn.ModuleList(GraphConv(self.view_feat[i], nhid, norm="none") for i in range(view_num)) 
        self.gc2 = GraphConv(nhid, nhid, norm="none") 
        self.dropout = dropout

    def forward(self, g, h):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        h : torch.Tensor
            features of nodes

        Returns
        -------
        x : torch.Tensor
            embedding of nodes
        """
        x = []
        for i in range(self.view_num):
            if i == self.view_num - 1:
                 x.append(h[:,i * (self.single_view):])
            else:
                x.append(h[:,i * self.single_view:(i + 1) * self.single_view])
        
        for i, gc in enumerate(self.gc1):
            if i != self.view_num - 1:
                x[i] = F.relu(self.gc1[i](g, x[i]))
            else:
                x[i] = F.relu(self.gc1[i](g, x[i]))
        
        for i in range(self.view_num):
            x[i] = F.dropout(x[i], self.dropout, training=self.training)
            x[i] = F.relu(self.gc2(g, x[i]))
        
        if self.agg_type == 1:
            rand_weight = torch.rand(1, self.view_num)
            for i in range(0, self.view_num):
                x[i] = rand_weight * x[i]
            x = torch.cat([i for i in x], 1)
        elif self.agg_type == 2:
            manual_weight = torch.tensor(self.agg_vec)
            for i in range(0, self.view_num):
                x[i] = manual_weight * x[i]
        
        x = torch.cat([i for i in x], 1)
        return x

class AttributeDecoder(nn.Module):
    """Attribute Decoder of DominantModel

    Parameters
    ----------
    nfeat : int
        dimension of feature 
    nhid : int
        dimension of hidden embedding
    dropout : float
        Dropout rate
    view_num : int
        number of view, by default:3
    """
    def __init__(self, nfeat, nhid, dropout, view_num):
        super(AttributeDecoder, self).__init__()
        self.gc1 = GraphConv(nhid * view_num, nhid,norm="none")
        self.gc2 = GraphConv(nhid, nfeat,norm="none")
        self.dropout = dropout

    def forward(self, g, h):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        h : torch.Tensor
            features of nodes

        Returns
        -------
        x : torch.Tensor
            Reconstructed attribute matrix
        """
        x = F.relu(self.gc1(g, h))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(g, x))
        return x

class StructureDecoder(nn.Module):
    """Structure Decoder of DominantModel

    Parameters
    ----------
    nhid : int
        dimension of hidden embedding
    dropout : float
        Dropout rate
    view_num : int
        number of view, by default: 3
    """
    def __init__(self, nhid, dropout, view_num):
        super(StructureDecoder, self).__init__()

        self.gc1 = GraphConv(nhid * view_num, nhid,norm="none")
        self.dropout = dropout

    def forward(self, g, h):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        h : torch.Tensor
            features of nodes

        Returns
        -------
        x : torch.Tensor
            Reconstructed adj matrix
        """
        x = F.relu(self.gc1(g, h))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x


class ALARMModel(nn.Module):
    """A Deep Multi-View Framework for Anomaly Detection on Attributed Networks.

    Parameters
    ----------
    feat_size : int
        dimension of feature 
    hidden_size : int
        dimension of hidden embedding (default: 64)
    dropout : float
        Dropout rate
    view_num : int
        number of view, by default 3
    agg_type: int
        aggregator type, by default 0. 0: Concatention | 1: Random aggregation weights | 2: Manual aggregation weights
    agg_vec : list
        weighted aggregation vector, bydefault: [1,1,1], stands for concatention. This is necessary if agg_type is 2.
    """
    def __init__(self, feat_size, hidden_size, dropout, view_num, agg_type, agg_vec):
        super(ALARMModel, self).__init__()
        if len(agg_vec) != view_num:
            raise KeyError('Aggregator vector size is {}, but the number of view is {}'.format(len(agg_vec), view_num))
        self.view_num = view_num
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout, view_num, agg_type, agg_vec)
        self.attr_decoder = AttributeDecoder(feat_size, hidden_size, dropout, view_num)
        self.struct_decoder = StructureDecoder(hidden_size, dropout, view_num)

    def forward(self, g, h):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        h : torch.Tensor
            features of nodes

        Returns
        -------
        struct_reconstructed : torch.Tensor
            Reconstructed adj matrix
        x_hat : torch.Tensor
            Reconstructed attribute matrix
        """
        # encode
        x = self.shared_encoder(g, h)
        # decode feature matrix
        x_hat = self.attr_decoder(g, x)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(g, x)
        # return reconstructed matrices
        return struct_reconstructed, x_hat

class ALARM(nn.Module):
    """A Deep Multi-View Framework for Anomaly Detection on Attributed Networks.
    
    
    Parameters
    ----------
    feat_size : int
        dimension of feature 
    hidden_size : int
        dimension of hidden embedding, by default: 64
    dropout : float
        Dropout rate
    view_num : int
        number of view, by default 3
    agg_type: int
        aggregator type, by default 0. 0: Concatention | 1: Random aggregation weights | 2: Manual aggregation weights
    agg_vec : list
        weighted aggregation vector, bydefault: [1,1,1], stands for concatention. This is necessary if agg_type is 2.
    """
    def __init__(self, feat_size, hidden_size, dropout, view_num, agg_type, agg_vec):
        super(ALARM, self).__init__()
        self.model = ALARMModel(feat_size, hidden_size, dropout, view_num, agg_type, agg_vec)
    
    def fit(self,graph,lr=5e-3,num_epoch=1,alpha=0.8,device='cpu'):
        """Fitting model

        Parameters
        ----------
        graph : dgl.DGLGraph
            graph dataset
        lr : float, optional
            learning rate, by default 5e-3
        num_epoch : int, optional
            number of training epochs , by default 1
        alpha : float, optional
            balance parameter, by default 0.8
        device : str, optional
            cuda id, by default 'cpu'
        """
        print('*'*20,'training','*'*20)

        features = graph.ndata['feat']
        print(features)
        adj = graph.adj(scipy_fmt='csr')

        adj_norm = normalize_adj(adj)
        adj_norm = torch.FloatTensor(adj_norm.toarray())

        import numpy as np
        print(np.sum(adj))
        adj_label = torch.FloatTensor(adj.toarray())
        
        print(graph)
        print('adj_label shape:', adj_label.shape)
        print('features shape:', features.shape)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            self.model = self.model.to(device)
            graph = graph.to(device)
            features = features.to(device)
            adj_label = adj_label.to(device)
            adj_norm = adj_norm.to(device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')      
 
        early_stop = EarlyStopping(early_stopping_rounds=10, patience=10)
        for epoch in range(num_epoch):
            
            loss, struct_loss, feat_loss = train_step(
                self.model, optimizer, graph, features,adj_label,alpha)
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
            )), "train/struct_loss=", "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()))

            early_stop(loss, self.model)
            if early_stop.isEarlyStopping():
                print(f"Early stopping in round {epoch}")
                break


    def predict(self, graph,alpha=0.8,device='cpu'):
        """predict and return anomaly score of each node

        Parameters
        ----------
        graph : dgl.DGLGraph
            graph dataset
        alpha : float, optional
            balance parameter, by default 0.8
        device : str, optional
            cuda id, by default 'cpu'

        Returns
        -------
        numpy.ndarray
            anomaly score of each node
        """
        print('*'*20,'predict','*'*20)

        features = graph.ndata['feat']
        adj = graph.adj(scipy_fmt='csr')

        adj_norm = normalize_adj(adj)
        adj_norm = torch.FloatTensor(adj_norm.toarray())        
        adj_label = torch.FloatTensor(adj.toarray())
       
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            graph = graph.to(device)
            features = features.to(device)
            adj_label = adj_label.to(device)
            adj_norm = adj_norm.to(device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')      
        
        predict_score = test_step(self.model, graph, features, adj_label,alpha)
   
        return predict_score