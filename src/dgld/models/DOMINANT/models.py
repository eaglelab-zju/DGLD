"""
Deep Anomaly Detection on Attributed Networks.[SDM19]
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
# from layers import GraphConvolution
from dgl.nn.pytorch import GraphConv
from .dominant_utils import get_parse, train_step, test_step,normalize_adj
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
    """
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid,norm="none")
        self.gc2 = GraphConv(nhid, nhid,norm="none")
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
        x = F.relu(self.gc1(g, h))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(g, x))

        return x

class Attribute_Decoder(nn.Module):
    """Attribute Decoder of DominantModel

    Parameters
    ----------
    nfeat : int
        dimension of feature 
    nhid : int
        dimension of hidden embedding
    dropout : float
        Dropout rate
    """
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()
        self.gc1 = GraphConv(nhid, nhid,norm="none")
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

class Structure_Decoder(nn.Module):
    """Structure Decoder of DominantModel

    Parameters
    ----------
    nhid : int
        dimension of hidden embedding
    dropout : float
        Dropout rate
    """
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()
        self.gc1 = GraphConv(nhid, nhid,norm="none")
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


class DominantModel(nn.Module):
    """Deep Anomaly Detection on Attributed Networks.[SDM19]

    Parameters
    ----------
    feat_size : int
        dimension of feature 
    hidden_size : int
        dimension of hidden embedding (default: 64)
    dropout : float
        Dropout rate
    """
    def __init__(self, feat_size, hidden_size, dropout):
        super(DominantModel, self).__init__()
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)

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

class Dominant(nn.Module):
    """Deep Anomaly Detection on Attributed Networks.[SDM19]
    ref:https://github.com/kaize0409/GCN_AnomalyDetection_pytorch
    
    Parameters
    ----------
    feat_size : int
        dimension of feature 
    hidden_size : int
        dimension of hidden embedding (default: 64)
    dropout : float
        Dropout rate
    
    """
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()
        self.model = DominantModel(feat_size, hidden_size, dropout)
    
    def fit(self,graph,lr=5e-3,logdir='tmp',num_epoch=1,alpha=0.8,device='cpu'):
        """Fitting model

        Parameters
        ----------
        graph : dgl.DGLGraph
            graph dataset
        lr : float, optional
            learning rate, by default 5e-3
        logdir : str, optional
            log dir, by default 'tmp'
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
        
        writer = SummaryWriter(log_dir=logdir)
        
        for epoch in range(num_epoch):
            
            loss, struct_loss, feat_loss = train_step(
                self.model, optimizer, graph, features,adj_label,alpha)
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
            )), "train/struct_loss=", "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()))
            writer.add_scalars(
                "loss",
                {"loss": loss, "struct_loss": struct_loss, "feat_loss": feat_loss},
                epoch,
            )
            writer.flush()


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