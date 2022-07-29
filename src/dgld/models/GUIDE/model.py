import torch 
import torch.nn as nn
import torch.nn.functional as F 
import dgl.nn as dglnn
import dgl.function as fn
import dgl 
import networkx as nx 
import numpy as np 
from tqdm import tqdm 
from networkx.generators.atlas import *
import sys 
import os
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from utils.evaluation import split_auc
from .guide_utils import get_struct_feat

from utils.early_stopping import EarlyStopping

class GUIDE():
    """
    Higher-order Structure Based Anomaly Detection on Attributed Networks
    2021 IEEE International Conference on Big Data

    Parameters
    ----------
    attrb_dim : int
        attributed feature dimensions of input data
    attrb_hid : int
        Hidden dimension for attribute autoencoder
    struct_dim : int
        struct feature dimensions of input data,you can use function get_struct_feat() to generate struct feature
        or use 6 as same as org paper 
    struct_hid : int
        Hidden dimension for struct autoencoder
    num_layers : int, optional
        Total number of layers. Default: 4.
    dropout : float, optional
        Dropout rate. Default: 0. .
    act : callable activation function, optional
        Activation function. Default: torch.nn.functional.relu

    Examples
    -------
    >>> gnd_dataset = GraphNodeAnomalyDectionDataset("Cora", p = 15, k = 50)
    >>> g = gnd_dataset[0]
    >>> label = gnd_dataset.anomaly_label
    >>> model = GUIDE(g.ndata['feat'].shape[1],256,6,64,num_layers=4,dropout=0.6)
    >>> model.fit(g,lr=0.001,num_epoch=200,device='0',alpha=0.9986,verbose=True,y_true=label)
    >>> result = model.predict(g,alpha=0.9986)
    >>> print(split_auc(label, result))
    """
    def __init__(self,attrb_dim,attrb_hid,struct_dim,struct_hid,num_layers=4,dropout=0,act=F.relu):
        self.num_layers = num_layers
        self.struct_feat = None
        self.model = GUIDEModel(attrb_dim,attrb_hid,struct_dim,struct_hid,num_layers,dropout=dropout,act=act)
        self.init_model()
    def init_model(self):
        """
        Initialize model
        """
        for weight in self.model.parameters():
            nn.init.xavier_normal_(weight.unsqueeze(0))
    def fit(self,graph,attrb_feat=None,struct_feat = None,lr=5e-3,batch_size=0,num_epoch=100,alpha=None,
            device='cpu',verbose=False,y_true=None):
        """
        train the model

        Parameters
        ----------
        graph : DGL.Graph
            input graph with feature named "feat" in g.ndata
        attrb_feat : torch.tensor, optional
            attribute feature,if don't set, auto use the graph.ndata['feat']. Default: None
        struct_feat : torch.tensor, optional
            struct feature,if don't set,auto generate by function get_struct_feat(). Default: None
        lr : float, optional
            learning rate for training. Default:5e-3
        logdir : str, optional
            tensorboard logdir. Default: 'tmp'
        num_epoch : int, optional
            number of epoch for training. Default: 100
        alpha : float, optional
            the weight about attribute loss and struct loss,if don't set, auto generate. Default: None
        device : str, optional
            device. Default : 'cpu' 
        verbose : bool, optional
            Verbosity mode. Turn on to print out log information. Default : False.
        y_true : list , optional
            The optional outlier ground truth labels used to monitor the training progress.
            Default : None
        """
        print('*'*20,'training','*'*20)
        if attrb_feat is None:
            attrb_feat = graph.ndata['feat']
        if struct_feat is None:
            struct_feat = get_struct_feat(graph)
            self.struct_feat = struct_feat
        if alpha is None:
            alpha = torch.std(struct_feat).detach() / (torch.std(struct_feat).detach() + torch.std(attrb_feat).detach())
        if batch_size == 0:
            # all data
            batch_size = attrb_feat.shape[0]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')    
        
        self.model = self.model.to(device)
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        # graph = graph.to(device)
        attrb_feat = attrb_feat.to(device)
        struct_feat = struct_feat.to(device)

        
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)


        nid =  torch.arange(graph.num_nodes())
        dataloader = dgl.dataloading.DataLoader(graph,nid, sampler,
        batch_size=batch_size, shuffle=True, drop_last=False)

        early_stop = EarlyStopping(early_stopping_rounds=10,patience=10)
        for epoch in range(num_epoch):
            self.model.train()
            epoch_loss = 0
            score = torch.zeros(graph.num_nodes())
            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [b.to(device) for b in blocks]
                attrb_feat_simple = attrb_feat[input_nodes]
                struct_feat_simple = struct_feat[input_nodes]
                attrb_feat_out = attrb_feat[output_nodes]
                struct_feat_out = struct_feat[output_nodes]
                attrb_rb,struct_rb = self.model(blocks,attrb_feat_simple,struct_feat_simple)
                loss = self.model.cal_loss(attrb_feat_out,attrb_rb,struct_feat_out,struct_rb,alpha)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if verbose:
                    if  y_true is not None:
                        score[output_nodes] = self.model.get_all_score(attrb_feat_out.detach(),attrb_rb.detach(),struct_feat_out.detach(),struct_rb.detach(),alpha).cpu()
            if verbose:
                print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(epoch_loss))
                if y_true is not None:
                    split_auc(y_true, score)
            early_stop(epoch_loss,self.model)
            if early_stop.isEarlyStopping():
                print(f"Early stopping in round {epoch}")
                break

    def predict(self,graph,attrb_feat=None,struct_feat = None,batch_size = 0,alpha=None,device='cpu'):
        """
        test the model
        
        Parameters
        ----------
        graph : DGL.Graph
            input graph with feature named "feat" in g.ndata
        attrb_feat : torch.tensor, optional
            attribute feature,if don't set, auto use the graph.ndata['feat']. Default: None
        struct_feat : torch.tensor, optional
            struct feature,if don't set,auto generate by function get_struct_feat(). Default: None
        alpha : float, optional
            the weight about attribute loss and struct loss,if don't set, auto generate. Default: None
        device : str, optional
            device. Default : 'cpu' 

        Returns
        -------
        score:torch.tensor
        the score of all nodes
        """
        print('*'*20,'predict','*'*20)
        if attrb_feat is None:
            attrb_feat = graph.ndata['feat']
        if struct_feat is None:
            if self.struct_feat is not None:
                struct_feat = self.struct_feat
            else:
                struct_feat = get_struct_feat(graph)
                self.struct_feat = struct_feat
        if alpha is None:
            alpha = torch.std(struct_feat).detach() / (torch.std(struct_feat).detach() + torch.std(attrb_feat).detach())
        if batch_size == 0:
            # all data
            batch_size = attrb_feat.shape[0]
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!') 
            
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        self.model = self.model.to(device)
        # graph = graph.to(device)
        attrb_feat = attrb_feat.to(device)
        struct_feat = struct_feat.to(device)
        
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)
        nid =  torch.arange(graph.num_nodes())
        dataloader = dgl.dataloading.DataLoader(graph,nid, sampler,
        batch_size=batch_size, shuffle=False, drop_last=False)

        self.model.eval()
        with torch.no_grad():
            score = torch.zeros(graph.num_nodes())
            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [b.to(device) for b in blocks]
                attrb_feat_simple = attrb_feat[input_nodes]
                struct_feat_simple = struct_feat[input_nodes]
                attrb_feat_out = attrb_feat[output_nodes]
                struct_feat_out = struct_feat[output_nodes]
                attrb_rb,struct_rb = self.model(blocks,attrb_feat_simple,struct_feat_simple)
                score[output_nodes] = self.model.get_all_score(attrb_feat_out.detach(),attrb_rb.detach(),struct_feat_out.detach(),struct_rb.detach(),alpha).cpu()
            return score


class GUIDEModel(nn.Module):
    def __init__(self,attrb_dim,attrb_hid,struct_dim,struct_hid,num_layers=4,dropout=0,act=F.relu):
        """
            GUIDE base model
            use the number of the moifts express the struct feature,through rebuild attribute feature and
            struct feature to anomaly detection
        Parameters
        ----------
        attrb_dim : int
            attributed feature dimensions of input data
        attrb_hid : int
            Hidden dimension for attribute autoencoder
        struct_dim : int
            struct feature dimensions of input data,you can use function get_struct_feat() to generate struct feature
            or use 6 same as org paper 
        struct_hid : int
            Hidden dimension for struct autoencoder
        num_layers : int, optional
            Total number of layers. Default: 4.
        dropout : float, optional
            Dropout rate. Default: 0. .
        act : callable activation function, optional
            Activation function. Default: torch.nn.functional.relu
        """
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.gna_layers = nn.ModuleList()
        self.gcn_layers.append(dglnn.GraphConv(attrb_dim,attrb_hid,allow_zero_in_degree=True,norm='both'))
        self.gna_layers.append(GNAConv(struct_dim,struct_hid))
        for i in range(num_layers-2):
            self.gcn_layers.append(dglnn.GraphConv(attrb_hid,attrb_hid,allow_zero_in_degree=True,norm='both'))
            self.gna_layers.append(GNAConv(struct_hid,struct_hid))
        self.gcn_layers.append(dglnn.GraphConv(attrb_hid,attrb_dim,allow_zero_in_degree=True,norm='both'))
        self.gna_layers.append(GNAConv(struct_hid,struct_dim))
        self.dropout = nn.Dropout(dropout)
        self.act = act 
    

    def forward(self,blocks,attr_feat,struct_feat):
        """
        The function to compute forward
        
        Parameters
        ----------
        blocks : list
            list of DGLBlock
        attrb_feat : torch.tensor
            attribute feature
        struct_feat : torch.tensor
            struct feature

        Returns
        -------
        attr_feat: torch.tensor
            rebuilded attribute feature
        struct_feat: torch.tensor
            rebuilded struct feature
        """
            
        for i,layer in enumerate(self.gcn_layers):
            with blocks[i].local_scope():
                attr_feat = layer(blocks[i],attr_feat)
                attr_feat = self.dropout(attr_feat)
                attr_feat = self.act(attr_feat)
        for i,layer in enumerate(self.gna_layers):
            with blocks[i].local_scope():
                struct_feat = layer(blocks[i],struct_feat)
                struct_feat = self.dropout(struct_feat)
                struct_feat = self.act(struct_feat)
        return attr_feat,struct_feat
                
    def cal_loss(self,attrb_org,attrb_rb,struct_org,struct_rb,alpha):
        """
        calculation the loss of the model

        Parameters
        ----------
        attrb_org:torch.tensor
            original attribute feature
        attrb_rb:torch.tensor
            rebuild attribute feature
        struct_org:torch.tensor
            original struct feature
        struct_rb:torch.tensor
            rebuild struct feature
        alpha : float, optional
            the weight about attribute loss and struct loss

        Returns
        -------
        loss : torch.tensor
            the loss of the model

        """
        attrb_diff = torch.pow(attrb_org-attrb_rb,2)
        struct_diff = torch.pow(struct_org-struct_rb,2)
        Ra = torch.sqrt(torch.sum(attrb_diff))
        Rs = torch.sqrt(torch.sum(struct_diff))
        loss = alpha * Ra + (1-alpha) * Rs 
        return loss 
    def get_all_score(self,attrb_org,attrb_rb,struct_org,struct_rb,alpha):
        """
        get all node's score

        Parameters
        ----------
        attrb_org:torch.tensor
            original attribute feature
        attrb_rb:torch.tensor
            rebuild attribute feature
        struct_org:torch.tensor
            original struct feature
        struct_rb:torch.tensor
            rebuild struct feature
        alpha : float, optional
            the weight about attribute loss and struct loss

        Returns
        -------
        score: torch.tensor
            shape is (num_node,1)
            the score of all nodes
        """
        attrb_diff = torch.pow(attrb_org-attrb_rb,2)
        struct_diff = torch.pow(struct_org-struct_rb,2)
        Ra = torch.sqrt(torch.sum(attrb_diff,dim=1))
        Rs = torch.sqrt(torch.sum(struct_diff,dim=1))
        score = alpha * Ra + (1-alpha) * Rs 
        return score

class GNAConv(nn.Module):
    """
    GNAConv for rebuild struct feature
    
    Parameters
    ----------
    in_feat: int
        input data dimension
    out_feat: int
        output data dimension
    """
    def __init__(self,in_feat,out_feat):
        super().__init__()
        self.w1 = nn.Linear(in_feat,out_feat)
        self.w2 = nn.Linear(in_feat,out_feat)
        self.a = nn.Parameter(torch.randn((1,out_feat)))
    def message_func(self,edges):
        """
        the message function

        Parameters
        ----------
        edges:dgl.DGLGraph.edges

        Returns
        -------
        msg : dict
            the message 
        """
        dt = edges.dst['h'] - edges.src['h']
        aij = torch.sum(self.a * self.w2(dt),dim=1)
        ss = self.w2(edges.src['h'])
        msg = {}
        msg['w_a'] = aij
        msg['val'] = ss
        return msg
    def forward(self,block,feat):
        """
        The function to compute forward
        
        Parameters
        ----------
        block : DGL.Graph.DGLBlock
            input graph
        feat : torch.tensor
            feature
        
        Returns
        -------
        out: torch.tensor
            out feature
        """
        with block.local_scope():
            h_src = feat
            h_dst = feat[:block.number_of_dst_nodes()]
            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst
            block.apply_edges(self.message_func)
            block.edata['w'] = dglnn.functional.edge_softmax(block,block.edata['w_a'])
            block.edata['val'] *= block.edata['w'].view(-1,1)
            block.update_all(fn.copy_e('val','val'),fn.sum('val','res'))
            out = block.dstdata['res']
            out += self.w1(block.dstdata['h'])
            return out