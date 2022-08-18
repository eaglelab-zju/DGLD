import torch
import torch.nn as nn

import dgl
from dgl.nn import GATConv

from copy import deepcopy
from .conad_utils import train_step, test_step, train_step_batch, test_step_batch

import sys
sys.path.append('../../')
from dgld.utils.early_stopping import EarlyStopping

class CONAD(nn.Module):
    """Contrastive Attributed Network Anomaly Detection with Data Augmentation.[PAKDD 2022]
    ref:https://github.com/zhiming-xu/conad
    
    Parameters
    ----------
    feat_size : int
        dimension of feature 
        
    Examples
    -------
    >>> from dgld.models.CONAD import CONAD
    >>> model = CONAD(feat_size=1433)
    >>> model.fit(g, num_epoch=1)
    >>> result = model.predict(g)
    """
    def __init__(self, 
                 feat_size
                 ):
        super(CONAD, self).__init__()
        self.model = CONAD_Base(feat_size)
        
    def fit(self,
            graph,
            lr=1e-3,
            weight_decay=0.,
            num_epoch=1,
            margin=0.5,
            alpha=0.9,
            eta=0.7,
            device='cpu',
            rate=0.2,
            contrast_type='siamese',
            batch_size=0,
            num_added_edge=50, 
            surround=50, 
            scale_factor=10
            ):
        """Fitting model

        Parameters
        ----------
        graph : dgl.DGLGraph
            graph dataset
        lr : float, optional
            learning rate, by default 1e-3
        weight_decay : float, optional
            weight decay (L2 penalty), by default 0.
        num_epoch : int, optional
            number of training epochs, by default 1
        margin : float, optional 
            parameter of the contrastive loss function, by default 0.5
        alpha : float, optional
            balance parameter, by default 0.9
        eta : float, optional
            balance parameter, by default 0.7
        device : str, optional
            device of computation, by default 'cpu'
        rate : float, optional
            the rate of anomalies, by default 0.2
        contrast_type : str, optional
            categories of contrastive loss function, by default 'siamese'
        batch_size : int, optional
            the size of training batch, by default 0
        num_added_edge : int
            parameter for generating high-degree anomalies, by default 50
        surround : int
            parameter for generating outlying anomalies, by default 50
        scale_factor : float
            parameter for generating disproportionate anomalies, by default 10    
        """
        print('*'*20,'training','*'*20)
        
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')
            
        if contrast_type == 'siamese':
            criterion = SiameseContrastiveLoss(margin=margin)
        elif contrast_type == 'triplet':
            criterion = TripletContrastiveLoss(margin=margin)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay) 
        
        graph = graph.remove_self_loop()
        
        g_orig = graph.add_self_loop()  
        transform = KnowledgeModel(rate=rate, num_added_edge=num_added_edge, surround=surround, scale_factor=scale_factor)
        g_aug = transform(deepcopy(graph)).add_self_loop() 
        
        self.model.to(device) 
        
        early_stop = EarlyStopping(early_stopping_rounds=10, patience=20)
        
        if batch_size == 0:
            print("full graph training!!!")
            for epoch in range(num_epoch):
                g_orig = g_orig.to(device)
                g_aug = g_aug.to(device)
                
                loss_epoch = train_step(self.model, optimizer, criterion, g_orig, g_aug, alpha=alpha, eta=eta)
                print("Epoch:", '%04d' % (epoch), "train/loss=", "{:.5f}".format(loss_epoch.item()))
                
                early_stop(loss_epoch.cpu().detach(), self.model)
                if early_stop.isEarlyStopping():
                    print(f"Early stopping in round {epoch}")
                    break
        else:
            print("batch graph training!!!")
            for epoch in range(num_epoch):
                loss = train_step_batch(self.model, optimizer, criterion, g_orig, g_aug, alpha, eta, batch_size, device)
                print("Epoch:", '%04d' % (epoch), "train/loss=", "{:.5f}".format(loss))    
                
                early_stop(loss, self.model)
                if early_stop.isEarlyStopping():
                    print(f"Early stopping in round {epoch}")
                    break     
            
            
    def predict(self,
                graph,
                alpha=0.9, 
                device='cpu',
                batch_size=0,
                ):
        """predict and return anomaly score of each node

        Parameters
        ----------
        graph : dgl.DGLGraph
            graph dataset
        alpha : float, optional
            balance parameter, by default 0.9
        device : str, optional
            device of computation, by default 'cpu'
        batch_size : int, optional
            the size of training batch, by default 0

        Returns
        -------
        numpy.ndarray
            anomaly score of each node
        """
        print('*'*20,'predict','*'*20)
        
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!') 
        
        if batch_size == 0:
            graph = graph.remove_self_loop().add_self_loop().to(device)
            self.model.to(device)
            predict_score = test_step(self.model, graph, alpha=alpha)
        else:
            graph = graph.remove_self_loop().add_self_loop()
            self.model.to(device)
            predict_score = test_step_batch(self.model, graph, alpha, batch_size, device)
            
        return predict_score
            

class SiameseContrastiveLoss(nn.Module):
    """siamese contrastive loss function
    
    Parameters
    ----------
    margin: float
        parameter of the contrastive loss function
    """
    def __init__(self, margin):
        super(SiameseContrastiveLoss, self).__init__()
        self.dist = lambda x, y: torch.linalg.norm(x-y, dim=1, ord=2)
        self.criterion = nn.TripletMarginLoss(margin=margin, reduction='none')
        
    def forward(self, z, z_hat, l, adj):
        """Forward Propagation

        Parameters
        ----------
        z : torch.Tensor
            feature representation of the original graph
        z_hat : torch.Tensor 
            feature representation for data augmentation graphs
        l : torch.Tensor
            anomaly labels
        adj : torch.Tensor 
            adjacency matrix

        Returns
        -------
        torch.Tensor
            the result of the loss function
        """
        l = l.view(-1)
        loss = self.dist(z, z_hat) * (l==0) + self.criterion(z, z, z_hat) * l
        return loss.mean()


class TripletContrastiveLoss(nn.Module):
    """triplet contrastive loss function
    
    Parameters
    ----------
    margin: float
        parameter of the contrastive loss function
    """
    def __init__(self, margin):
        super(TripletContrastiveLoss, self).__init__()
        self.criterion = nn.TripletMarginLoss(margin=margin, reduction='sum')
        
    def forward(self, orig, aug, l, adj):
        """Forward Propagation

        Parameters
        ----------
        orig : torch.Tensor
            feature representation of the original graph
        aug : torch.Tensor 
            feature representation for data augmentation graphs
        l : torch.Tensor
            anomaly labels
        adj : torch.Tensor 
            adjacency matrix

        Returns
        -------
        torch.Tensor
            the result of the loss function
        """
        l = l.view(-1)
        adj = torch.diag(l==0).float() @ adj @ torch.diag(l==1).float()
        pairs = torch.nonzero(adj)
        loss = self.criterion(orig[pairs[:, 0]], orig[pairs[:, 1]], aug[pairs[:, 1]])
        return loss
        
        
class CONAD_Base(nn.Module):
    """This is a basic structure model of CONAD.

    Parameters
    ----------
    in_feats : int
        the feature dimension of the input data.
    hid_feats : int, optional
        the dimension of hidden feature, by default 128    
    out_feats : int, optional
        the dimension of output feature, by default 64    
    num_heads: int, optional
        number of heads in Multi-Head Attention.
    activation: callable activation function/layer or None, optional
        if not None, applies an activation function to the updated node features, by default nn.LeakyReLU()
    """
    def __init__(self,
                 in_feats,
                 hid_feats=128,
                 out_feats=64,
                 num_heads=2,
                 activation=nn.LeakyReLU()
                 ):
        super(CONAD_Base, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        
        self.shared_encoder = nn.ModuleList([
            GATConv(in_feats, hid_feats, num_heads, activation=activation),
            GATConv(hid_feats*num_heads, out_feats, num_heads, activation=activation),
        ])
        
        self.attr_decoder = GATConv(out_feats, in_feats, num_heads, activation=activation)
        
        self.struct_decoder = lambda h: h @ h.T 
        # self.struct_decoder = lambda h: torch.sigmoid(h @ h.T) 
        
    def embed(self, g, h):
        """compute embeddings 

        Parameters
        ----------
        g : dgl.DGLGraph or list
            graph dataset
        h : torch.Tensor
            features of nodes

        Returns
        -------
        torch.Tensor
            embeddings of nodes
        """
        if isinstance(g, list):
            return self.embed_batch(g, h)
        for i, layer in enumerate(self.shared_encoder):
            h = layer(g, h)
            h = h.flatten(1) if i == 0 else h.mean(1)
        return h
    
    def reconstruct(self, g, h):
        """reconstruct attribute matrix and adjacency matrix

        Parameters
        ----------
        g : dgl.DGLGraph or list
            graph dataset
        h : torch.Tensor
            feature representation of nodes

        Returns
        -------
        torch.Tensor 
            reconstructed adjacency matrix
        torch.Tensor 
            reconstructed attribute matrix
        """
        if isinstance(g, list):
            return self.reconstruct_batch(g, h)
        # reconstruct attribute matrix
        x_hat = self.attr_decoder(g, h).mean(1)
        # reconstruct adjacency matrix
        a_hat = self.struct_decoder(h)
        return a_hat, x_hat
    
    def forward(self, g, h):
        """Forward Propagation

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        h : torch.Tensor or list
            feature representation of nodes

        Returns
        -------
        torch.Tensor 
            reconstructed adjacency matrix
        torch.Tensor 
            reconstructed attribute matrix
        """
        if isinstance(g, list):
            return self.forward_batch(g, h)
        # compute embeddings 
        h = self.embed(g, h)
        # reconstruct
        a_hat, x_hat = self.reconstruct(g, h)
        return a_hat, x_hat 
        
    def embed_batch(self, blocks, h):
        """compute embeddings for mini-batch graph training

        Parameters
        ----------
        g : list
            graph dataset
        h : torch.Tensor
            features of nodes

        Returns
        -------
        torch.Tensor
            embeddings of nodes
        """
        for i, layer in enumerate(self.shared_encoder):
            h = layer(blocks[i], h)
            h = h.flatten(1) if i == 0 else h.mean(1)
        return h
    
    def reconstruct_batch(self, blocks, h):
        """reconstruct attribute matrix and adjacency matrix for mini-batch graph training

        Parameters
        ----------
        g : list
            graph dataset
        h : torch.Tensor
            feature representation of nodes

        Returns
        -------
        torch.Tensor 
            reconstructed adjacency matrix
        torch.Tensor 
            reconstructed attribute matrix
        """
        # reconstruct attribute matrix
        x_hat = self.attr_decoder(blocks[-1], h).mean(1)
        # reconstruct adjacency matrix
        a_hat = self.struct_decoder(h)
        return a_hat, x_hat
    
    def forward_batch(self, blocks, h):
        """Forward Propagation for mini-batch graph training

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset
        h : list
            feature representation of nodes

        Returns
        -------
        torch.Tensor 
            reconstructed adjacency matrix
        torch.Tensor 
            reconstructed attribute matrix
        """
        # compute embeddings 
        h = self.embed(blocks, h)
        # reconstruct
        a_hat, x_hat = self.reconstruct(blocks, h)
        return a_hat, x_hat         
        
            
class KnowledgeModel(dgl.BaseTransform):
    """Knowledge Modeling Module, introduce a certain amount of anomalies belonging to each anomaly type to the input attributed network to form an augmented attributed network

    Parameters
    ----------
    rate : float
        rate for generating anomalies
    num_added_edge : int
        parameter for generating high-degree anomalies
    surround : int
        parameter for generating outlying anomalies
    scale_factor : float
        parameter for generating disproportionate anomalies

    """
    def __init__(self, rate, num_added_edge, surround, scale_factor):
        self.rate = rate
        self.num_added_edge = num_added_edge
        self.surround = surround
        self.scale_factor = scale_factor
        
    def __call__(self, g, ntype='feat'):
        with g.local_scope():
            adj = g.adj().to_dense().detach()
            feat = g.ndata[ntype].detach()
            num_nodes = adj.shape[0]
            label = torch.zeros(num_nodes)
            
            # aug or not
            prob = torch.rand_like(label)
            label[prob < self.rate] = 1
            
            # high-degree
            num_hd = torch.sum(prob < self.rate / 4)
            edge_mask = torch.rand(num_hd, num_nodes) < self.num_added_edge / num_nodes
            adj[prob <= self.rate / 4, :] = edge_mask.float()
            adj[:, prob <= self.rate / 4] = edge_mask.float().T
            
            # outlying
            ol_mask = torch.logical_and(self.rate / 4 <= prob, prob < self.rate / 2)
            adj[ol_mask, :] = 0
            adj[:, ol_mask] = 0
            
            # deviated
            dv_mask = torch.logical_and(self.rate / 2 <= prob, prob < self.rate * 3 / 4)
            feat_c = feat[torch.randperm(num_nodes)[:self.surround]]
            ds = torch.cdist(feat[dv_mask], feat_c)
            feat[dv_mask] = feat_c[torch.argmax(ds, 1)]
            
            # disproportionate
            mul_mask = torch.logical_and(self.rate * 3 / 4 <= prob, prob < self.rate * 7 / 8)
            div_mask = self.rate * 7 / 8 <= prob
            feat[mul_mask] *= self.scale_factor
            feat[div_mask] /= self.scale_factor
            
            g_aug = dgl.graph(adj.nonzero(as_tuple=True), num_nodes=num_nodes)
            g_aug.ndata['feat'] = feat
            g_aug.ndata['label'] = label
            return g_aug        