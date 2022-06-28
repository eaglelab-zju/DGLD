"""
AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks
"""
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv  # ,GATConv1
from torch import nn
import dgl
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .anomalydae_utils import train_step, test_step,normalize_adj


class AnomalyDAE(nn.Module):
    """AnomalyDAE: Dual autoencoder for anomaly detection on attributed networks
    ref:https://github.com/haoyfan/AnomalyDAE
    
    Parameters
    ----------
    feat_size : int
        dimension of feature 
    num_nodes : int
        number of nodes
    embed_dim : int
        dimension of hidden embedding (default: 256)
    out_dim : int
        dimension of output embedding (default: 128)
    dropout : float
        Dropout rate
    
    """
    def __init__(self, feat_size,num_nodes,embed_dim,out_dim, dropout):
        super(AnomalyDAE, self).__init__()
        self.model = AnomalyDAEModel(in_feat_dim=feat_size,in_num_dim=num_nodes,embed_dim=embed_dim,
                        out_dim=out_dim,dropout=dropout)
    
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
        print(features)
        adj = graph.adj(scipy_fmt='csr')

        print(np.sum(adj))
        adj_label = torch.FloatTensor(adj.toarray())
        
        print(graph)
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
    
        writer = SummaryWriter(log_dir=logdir)
        
        for epoch in range(num_epoch):
            loss, struct_loss, feat_loss,_ = train_step(
            self.model, optimizer, graph, features,adj_label,alpha,eta,theta)
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
            )), "train/struct_loss=", "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()))
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
        adj_label = torch.FloatTensor(adj.toarray())
       
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            graph = graph.to(device)
            features = features.to(device)
            adj_label = adj_label.to(device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')      
        
        predict_score = test_step(self.model, graph, features, adj_label, alpha,eta,theta)
   
        return predict_score

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


class AnomalyDAEModel(nn.Module):
    """
    AdnomalyDAE is an anomaly detector consisting of a structure autoencoder,
    and an attribute reconstruction autoencoder. 

    Parameters
    ----------
    in_feat_dim : int
         Dimension of input feature
    in_num_dim: int
         Dimension of the input number of nodes
    embed_dim: int
         Dimension of the embedding after the first reduced linear layer (D1)   
    out_dim : int
         Dimension of final representation
    dropout : float, optional
        Dropout rate of the model
        Default: 0

    """
    def __init__(self,
                 in_feat_dim,
                 in_num_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 ):
        super(AnomalyDAEModel, self).__init__()
        self.structure_AE = StructureAE(in_feat_dim,in_num_dim, embed_dim,
                                        out_dim, dropout)
        self.attribute_AE = AttributeAE(in_num_dim, embed_dim,
                                        out_dim, dropout)

    def forward(self, g, x):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        x : torch.Tensor
            features of nodes

        Returns
        -------
        A_hat : torch.Tensor
            Reconstructed adj matrix
        X_hat : torch.Tensor
            Reconstructed attribute matrix

        """
        A_hat, embed_x = self.structure_AE(g, x)
        X_hat = self.attribute_AE(x, embed_x)
        return A_hat, X_hat


class StructureAE(nn.Module):
    """
    Structure Autoencoder in AnomalyDAE model: the encoder
    transforms the node attribute X into the latent
    representation with the linear layer, and a graph attention
    layer produces an embedding with weight importance of node 
    neighbors. Finally, the decoder reconstructs the final embedding
    to the original.
    See :cite:`fan2020anomalydae` for details.

    Parameters
    ----------
    in_dim: int
        input dimension of node data
    in_num_dim: int
        number of nodes
    embed_dim: int
        the latent representation dimension of node
       (after the first linear layer)
    out_dim: int 
        the output dim after the graph attention layer
    dropout: float
        dropout probability for the linear layer

    """

    def __init__(self,
                 in_dim,
                 in_num_dim,
                 embed_dim,
                 out_dim,
                 dropout):
        super(StructureAE, self).__init__()
        self.dense = nn.Linear(in_dim, embed_dim)  # (n,feat_dim)->(n,emd_dim)
        # self.attention_layer = GATConv(embed_dim, out_dim, num_heads=1)
        self.attention_layer = NodeAttention(embed_dim,nb_nodes=in_num_dim,
                                       # act=tf.nn.relu,
                                       act=lambda x: x,
                                       out_sz=out_dim)
        self.dropout = dropout
        for module in self.modules():
            init_weights(module)

    def forward(self, g, x):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        x : torch.Tensor
            features of nodes

        Returns
        -------
        x : torch.Tensor
            Reconstructed attribute (feature) of nodes.
        embed_x : torch.Tensor
            Embedd nodes after the attention layer

        """
        # encoder
        x = torch.relu(self.dense(x))
        x = F.dropout(x, self.dropout)
        embed_x = self.attention_layer(x,g.adj())  # (n,128)
        # print(embed_x)
        # decoder
        x = torch.sigmoid(embed_x @ embed_x.T)  # (n,n)
        # x=torch.relu(embed_x @ embed_x.T)
        return x, embed_x

class NodeAttention(nn.Module):
    """node attention layer

    Parameters
    ----------
    embed_dim : int
        the latent representation dimension of node
    out_sz : int
        the output dim after the graph attention layer
    nb_nodes : int
        number of nodes
    dropout : float, optional
        dropout probability for the linear layer, by default 0.
    act : F, optional
        Choice of activation function , by default F.elu

    """
    def __init__(self,embed_dim, out_sz, nb_nodes, dropout=0.,
                 act=F.elu, **kwargs):
        
        super(NodeAttention, self).__init__(**kwargs)
        self.act = act
        self.out_sz = out_sz
        self.bias_mat = None
        self.nb_nodes = nb_nodes
        self.conv1=torch.nn.Conv1d(embed_dim, self.out_sz, 1, bias=False)
        self.conv2=torch.nn.Conv1d(self.out_sz, 1, 1)
        self.conv3=torch.nn.Conv1d(self.out_sz, 1, 1)


    def forward(self,inputs,adj):
        """Forward Propagation

        Parameters
        ----------
        inputs : torch.Tensor
            features of nodes
        adj : numpy.matrix
            adj matrix

        Returns
        -------
        torch.Tensor
            Embedd nodes after the attention layer
        """
        self.bias_mat = adj.to_dense().to(inputs.device)
        inputs=torch.unsqueeze(inputs,1).permute(0,2,1)
        seq_fts = self.conv1(inputs)

        # simplest self-attention possible
        
        f_1_t = self.conv2(seq_fts)
        f_2_t = self.conv3(seq_fts)
        # print('nb_nodes',self.nb_nodes)
        f_1 = torch.reshape(f_1_t, (self.nb_nodes, 1))
        f_2 = torch.reshape(f_2_t, (self.nb_nodes, 1))
        # print(' self.bias_mat shape', self.bias_mat.shape)
        # print(' f_1 shape', f_1.shape)
        
        f_1 = self.bias_mat * f_1
        f_2 = self.bias_mat * f_2.T


        logits = torch.add(f_1, f_2)
        lrelu = F.leaky_relu(logits)
        
        coefs = torch.softmax(lrelu,dim=1)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = torch.reshape(coefs, [self.nb_nodes, self.nb_nodes])
        seq_fts = torch.squeeze(seq_fts)
        vals = coefs @ seq_fts
        
        return vals  # activation

class AttributeAE(nn.Module):
    """
    Attribute Autoencoder in AnomalyDAE model: the encoder
    employs two non-linear feature transform to the node attribute
    x. The decoder takes both the node embeddings from the structure
    autoencoder and the reduced attribute representation to 
    reconstruct the original node attribute.

    Parameters
    ----------
    in_dim:  int
        dimension of the input number of nodes
    embed_dim: int
        the latent representation dimension of node
        (after the first linear layer)
    out_dim:  int
        the output dim after two linear layers
    dropout: float
        dropout probability for the linear layer  
         
   
    """

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 dropout):
        super(AttributeAE, self).__init__()
        # (feat_dim,n)->(feat_dim,emd_dim)
        self.dense1 = nn.Linear(in_dim, embed_dim)
        self.dense2 = nn.Linear(embed_dim, out_dim)
        self.dropout = dropout
        for module in self.modules():
            init_weights(module)

    def forward(self,
                x,
                struct_embed):
        """Forward Propagation

        Parameters
        ----------
        x : torch.Tensor
            features of nodes
        struct_embed : torch.Tensor
            Embedd nodes after the attention layer

        Returns
        -------
        x : torch.Tensor
            Reconstructed attribute (feature) of nodes.
        """
        # encoder
        x = torch.relu(self.dense1(x.T))  # (d,256)
        x = F.dropout(x, self.dropout)
        x = torch.relu(self.dense2(x))  # (d,128)
        x = F.dropout(x, self.dropout)
        # decoder
        x = struct_embed @ x.T  # (n,128) * (128,d)
        return x  # (n,d)
