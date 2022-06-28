"""
ComGA: Community-Aware Attributed Graph Anomaly Detection
"""
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch import nn
import dgl
import scipy.sparse as sp
import scipy.io as sio
from sklearn.metrics import precision_score, roc_auc_score
import networkx as nx
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .comga_utils import train_step, test_step,normalize_adj


class ComGA(nn.Module):
    """ComGA: Community-Aware Attributed Graph Anomaly Detection
    ref:https://github.com/DASE4/ComGA
    
    Parameters
    ----------
    num_nodes : int
        number of nodes
    num_feats : int
        dimension of feature 
    n_enc_1 : int
        number of encode1 units
    n_enc_2 : int
        number of encode2 units
    n_enc_3 : int
        number of encode3 units
    dropout : float
        Dropout rate
    
    """
    def __init__(self,num_nodes,num_feats,n_enc_1,n_enc_2,n_enc_3, dropout):
        super(ComGA, self).__init__()
        self.model = ComGAModel(num_nodes=num_nodes,num_feats=num_feats,
                        n_enc_1=n_enc_1,n_enc_2=n_enc_2,n_enc_3=n_enc_3,dropout=dropout)
    
    def fit(self,graph,lr=5e-3,logdir='tmp',num_epoch=1,alpha=0.7,eta=5.0,theta=40.0,device='cpu'):
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
        eta : float, optional
            Attribute penalty balance parameter, by default 5.0
        theta : float, optional
            structure penalty balance parameter, by default 40.0
        device : str, optional
            cuda id, by default 'cpu'
        """
        print('*'*20,'training','*'*20)

        features = graph.ndata['feat']
        adj = graph.adj(scipy_fmt='csr')

        A=adj.toarray()
        k1 = np.sum(A, axis=1)
        k2 = k1.reshape(k1.shape[0], 1)
        k1k2 = k1 * k2
        num_loop=0
        for i in range(adj.shape[0]):
            if adj[i,i]==1:
                num_loop+=1
        m=(np.sum(A)-num_loop)/2+num_loop
        Eij = k1k2 / (2 * m)
        B =np.array(A - Eij)

        print(np.sum(adj))
        adj_label = torch.FloatTensor(adj.toarray())
        B = torch.FloatTensor(B)
        
        print(graph)
        print('B shape:', B.shape)
        print('adj_label shape:', adj_label.shape)
        print('features shape:', features.shape)
        

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')      

        self.model = self.model.to(device)
        graph = graph.to(device)
        features = features.to(device)
        adj_label = adj_label.to(device)
        B = B.to(device)
    
        writer = SummaryWriter(log_dir=logdir)
        
        for epoch in range(num_epoch):
            loss,struct_loss, feat_loss,kl_loss,re_loss,_ = train_step(
                self.model, optimizer, graph, features, B,adj_label,alpha,eta,theta,device)
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
                    )),"train/kl_loss=", "{:.5f}".format(kl_loss.item()),
                    "train/struct_loss=", "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()),
            )
            writer.add_scalars(
                "loss",
                {"loss": loss, "struct_loss": struct_loss, "feat_loss": feat_loss},
                epoch,
            )
            writer.flush()


    def predict(self, graph, alpha=0.7,eta=5.0,theta=40.0,device='cpu'):
        """predict and return anomaly score of each node

        Parameters
        ----------
        graph : dgl.DGLGraph
            graph dataset
        alpha : float, optional
            balance parameter, by default 0.8
        eta : float, optional
            Attribute penalty balance parameter, by default 5.0
        theta : float, optional
            structure penalty balance parameter, by default 40.0
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

        A=adj.toarray()
        k1 = np.sum(A, axis=1)
        k2 = k1.reshape(k1.shape[0], 1)
        k1k2 = k1 * k2
        num_loop=0
        for i in range(adj.shape[0]):
            if adj[i,i]==1:
                num_loop+=1
        m=(np.sum(A)-num_loop)/2+num_loop
        Eij = k1k2 / (2 * m)
        B =np.array(A - Eij)

        print(np.sum(adj))
        adj_label = torch.FloatTensor(adj.toarray())
        B = torch.FloatTensor(B)
        
        print(graph)
        print('B shape:', B.shape)
        print('adj_label shape:', adj_label.shape)
        print('features shape:', features.shape)

        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')      

        self.model = self.model.to(device)
        graph = graph.to(device)
        features = features.to(device)
        adj_label = adj_label.to(device)
        B = B.to(device)
        predict_score = test_step(self.model, graph, features, B,adj_label, alpha, eta, theta,device)
        return predict_score

class ComGAModel(nn.Module):
    """
    ComGA is an anomaly detector consisting of a Community Detection Module,
    a tGCN Module and an Anomaly Detection Module

    Parameters
    ----------
    num_nodes : int
        number of nodes
    num_feats : int
        dimension of feature 
    n_enc_1 : int
        number of encode1 units
    n_enc_2 : int
        number of encode2 units
    n_enc_3 : int
        number of encode3 units
    dropout : float
        Dropout rate
    """

    def __init__(self,
                 num_nodes,
                 num_feats,
                 n_enc_1, n_enc_2, n_enc_3,
                 dropout,
                 ):
        super(ComGAModel, self).__init__()
        self.commAE=CommunityAE(num_nodes,n_enc_1, n_enc_2, n_enc_3,dropout)
        self.tgcnEnc=tGCNEncoder(num_feats,n_enc_1, n_enc_2, n_enc_3,dropout)
        self.attrDec=AttrDecoder(num_feats,n_enc_1, n_enc_2, n_enc_3,dropout)
        self.struDec=StruDecoder(dropout)
        

    def forward(self,g,x,B):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        x : torch.Tensor
            features of nodes
        B : torch.Tensor
            Modularity Matrix

        Returns
        -------
        A_hat : torch.Tensor
            Reconstructed adj matrix
        X_hat : torch.Tensor
            Reconstructed attribute matrix
        B_hat : torch.Tensor
            Reconstructed Modularity Matrix
        z : torch.Tensor
            the node latent representation of tgcnEnc
        z_a : torch.Tensor
            the node latent representation of CommunityAE
        """
        B_enc1,B_enc2,z_a,B_hat = self.commAE(B)
        z = self.tgcnEnc(g,x,B_enc1,B_enc2,z_a)
        X_hat = self.attrDec(g,z)
        A_hat = self.struDec(z)
        
        return A_hat,X_hat,B_hat,z,z_a

def init_weights(module: nn.Module) -> None:
    """Init Module Weights
    ```python
        for module in self.modules():
            init_weights(module)
    ```
    Parameters
    ----------
    module : nn.Module

    """
    if isinstance(module, nn.Linear):
        # TODO: different initialization
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Bilinear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

class CommunityAE(nn.Module):
    """
    Community Detection Module:
        The modularity matrix B is reconstructed by autoencode to obtain a representation 
        of each node with community information.
    
    Parameters
    ----------
    num_nodes : int
        number of nodes
    n_enc_1 : int
        number of encode1 units
    n_enc_2 : int
        number of encode2 units
    n_enc_3 : int
        number of encode3 units
    dropout : float
        Dropout rate

    """

    def __init__(self,
                 num_nodes,
                 n_enc_1, n_enc_2, n_enc_3,
                 dropout):
        super(CommunityAE, self).__init__()
        self.dropout=dropout
        #encoder
        self.enc1 = nn.Linear(num_nodes, n_enc_1)
        self.enc2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc3 = nn.Linear(n_enc_2, n_enc_3)
        #decoder
        self.dec1 = nn.Linear(n_enc_3, n_enc_2)
        self.dec2 = nn.Linear(n_enc_2, n_enc_1)
        self.dec3 = nn.Linear(n_enc_1, num_nodes)

        for module in self.modules():
            init_weights(module)

    def forward(self,B):
        """Forward Propagation

        Parameters
        ----------
        B : torch.Tensor
            Modularity Matrix

        Returns
        -------
        hidden1 : torch.Tensor
            the node latent representation of encoder1
        hidden2 : torch.Tensor
            the node latent representation of encoder2
        z_a : torch.Tensor
            the node latent representation of encoder3
        community_reconstructions : torch.Tensor
            Reconstructed Modularity Matrix
        """
        # encoder
        x = torch.relu(self.enc1(B))
        hidden1 = F.dropout(x, self.dropout)
        x=torch.relu(self.enc2(hidden1))
        hidden2 = F.dropout(x, self.dropout)
        x=torch.relu(self.enc3(hidden2))
        z_a = F.dropout(x, self.dropout)
        
        # decoder
        x=torch.relu(self.dec1(z_a))
        se1 = F.dropout(x, self.dropout)
        x=torch.relu(self.dec2(se1))
        se2 = F.dropout(x, self.dropout)
        x=torch.sigmoid(self.dec3(se2))
        community_reconstructions = F.dropout(x, self.dropout)
        
        return hidden1,hidden2,z_a,community_reconstructions


class tGCNEncoder(nn.Module):
    """
    tGCNEncoder:
        To effectively fuse community structure information to GCN model for structure anomaly,
    and learn more distinguishable anomalous node representations for local, global, and 
    structure anomalies.

    Parameters
    ----------
    in_feats : int
        dimension of feature 
    n_enc_1 : int
        number of encode1 units
    n_enc_2 : int
        number of encode2 units
    n_enc_3 : int
        number of encode3 units
    dropout : float
        Dropout rate
    
    """

    def __init__(self,
                 in_feats,
                 n_enc_1, n_enc_2, n_enc_3,
                 dropout):
        super(tGCNEncoder, self).__init__()
        self.enc1=GraphConv(in_feats,n_enc_1,activation=F.relu)
        self.enc2=GraphConv(n_enc_1,n_enc_2,activation=F.relu)
        self.enc3=GraphConv(n_enc_2,n_enc_3,activation=F.relu)
        self.enc4=GraphConv(n_enc_3,n_enc_3,activation=F.relu)

        self.dropout = dropout
        for module in self.modules():
            init_weights(module)

    def forward(self,g,x,B_enc1,B_enc2,B_enc3):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        x : torch.Tensor
            features of nodes
        B_enc1 : torch.Tensor
            the node latent representation of CommunityAE encoder1
        B_enc2 : torch.Tensor
            the node latent representation of CommunityAE encoder2
        B_enc3 : torch.Tensor
            the node latent representation of CommunityAE encoder3

        Returns
        -------
        torch.Tensor
            the node latent representation of tGCNEncoder
        """
        # encoder
        x1=F.dropout(self.enc1(g,x), self.dropout)
        x=x1+B_enc1
        x2=F.dropout(self.enc2(g,x), self.dropout)
        x=x2+B_enc2
        x3=F.dropout(self.enc3(g,x), self.dropout)
        x=x3+B_enc3
        z=F.dropout(self.enc4(g,x), self.dropout)

        return z


class AttrDecoder(nn.Module):
    """
    AttrDecoder:
        utilize attribute decoder to take the learned latent representation
    Z as input to decode them for reconstruction of original nodal attributes.

    Parameters
    ----------
    in_feats : int
        dimension of feature 
    n_enc_1 : int
        number of encode1 units
    n_enc_2 : int
        number of encode2 units
    n_enc_3 : int
        number of encode3 units
    dropout : float
        Dropout rate

    """

    def __init__(self,
                 in_feats,
                 n_enc_1, n_enc_2, n_enc_3,
                 dropout):
        super(AttrDecoder, self).__init__()
        self.dropout=dropout

        self.dec1=GraphConv(n_enc_3,n_enc_3,activation=F.relu)
        self.dec2=GraphConv(n_enc_3,n_enc_2,activation=F.relu)
        self.dec3=GraphConv(n_enc_2,n_enc_1,activation=F.relu)
        self.dec4=GraphConv(n_enc_1,in_feats,activation=F.relu)
        
        for module in self.modules():
            init_weights(module)

    def forward(self,g,z):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        z : torch.Tensor
            the node latent representation

        Returns
        -------
        torch.Tensor
            Reconstructed attribute matrix
        """
        # decoder
        x1=F.dropout(self.dec1(g,z), self.dropout)
        x2=F.dropout(self.dec2(g,x1), self.dropout)
        x3=F.dropout(self.dec3(g,x2), self.dropout)
        attribute_reconstructions=F.dropout(self.dec4(g,x3), self.dropout)
        
        return attribute_reconstructions    


class StruDecoder(nn.Module):
    """
    StruDecoder:
        utilize structure decoder to take the learned latent representation 
    Z as input to decode them for reconstruction of original graph structure.

    Parameters
    ----------
    dropout : float
        Dropout rate

    """

    def __init__(self,
                 dropout):
        super(StruDecoder, self).__init__()
        self.dropout=dropout

    def forward(self,z):
        """Forward Propagation

        Parameters
        ----------
        z : torch.Tensor
            the node latent representation

        Returns
        -------
        torch.Tensor
            Reconstructed adj matrix
        """
        # decoder
        x=F.dropout(z, self.dropout)
        x=z@x.T
        structure_reconstructions=torch.sigmoid(x)

        return structure_reconstructions    






