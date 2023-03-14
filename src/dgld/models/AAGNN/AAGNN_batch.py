"""
This is a model for large graph training based on the AAGNN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.early_stopping import EarlyStopping
import dgl.function as fn
import dgl 
class AAGNN_batch(nn.Module):
    """
    This is a model for large graph training based on the AAGNN model.

    Parameters
    ----------
    feat_size : int
        The feature dimension of the input data.
    out_feats : int
        The dimension of output feature.
    """
    def __init__(self, feat_size, out_feats=300):
        super().__init__()
        self.model = model_base(feat_size, out_feats)
        self.out_feats = out_feats
    
    def fit(self, graph, num_epoch=100, device='cpu', lr=0.001,weight_decay=0.001):
        """
        This is a function used to train the model.

        Parameters
        ----------
        graph : dgl
            The graph data you input.
        
        num_epoch : int
            The number of times you want to train the model.
        
        device : str
            The number of times you want to train the model.
        
        lr : float
            Learning rate for model training.
        
        weight_decay : float
            L2 weight_decay.

        """
        
        print('-'*40, 'training', '-'*40)
        if device != 'cpu':
            device = 'cuda:' + device

        print('device=',device)
        graph = graph.to(device)
        self.model.to(device)
        features = graph.ndata['feat']

        opt = torch.optim.Adam(self.model.parameters(), lr=lr,weight_decay=weight_decay)
        
        self.center,node_ids = self.cal_center(graph, features, 0.5)
        early_stop = EarlyStopping(early_stopping_rounds=10, patience=10)
        
        for epoch in range(num_epoch):
            self.model.train()
            h = self.model(graph,features)
            loss = self.loss_fun(h[node_ids],self.center)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f'epoch:{epoch:04d} loss is {loss.item():.5f}')
            early_stop(loss)
            if early_stop.isEarlyStopping():
                print(f"Early stopping in round{epoch}")
                break

    def predict(self, graph, device='cpu'):
        """
        This is a function that loads a trained model for predicting graph data.

        Parameters
        ----------
        graph : dgl
            The graph data you input.
        
        device : str
            The number of times you want to train the model.
        
        subgraph_size : int
            The size of training subgraph.

        Returns
        -------
        score : numpy
            A vector of decimals representing the anomaly score for each node.
        """

        print('-'*40, 'predicting', '-'*40)
        if device != 'cpu':
            device = 'cuda:' + device
            
        self.model = self.model.to(device)
        features = graph.ndata['feat']
        graph = graph.to(device)
        features = features.to(device)
        self.model.eval()
        with torch.no_grad():
            h = self.model(graph,features)
            score = torch.sum(torch.pow(h-self.center,2),dim=-1).detach().cpu()
        
        return score.numpy()

    def cal_center(self, graph, x, rate):
        """
        This is a function to calculate the center vector.

        Parameters
        ----------
        graph : dgl
            The graph data you input.
        
        model : tensor
            The model we trained.

        device : str
            The number of times you want to train the model.
        
        subgraph_size : int
            The size of training subgraph.
        
        edge_dic: dict
            The input node-to-node relationship dictionary.


        Returns
        -------
        center : numpy.ndarray
            The center vector of all samples.
        """
        self.model.eval()
        with torch.no_grad():
            h = self.model(graph,x)
            center = torch.mean(h,dim=0).detach()
            dis = torch.sum(torch.pow(center.repeat(h.shape[0],1)-h,2),dim=-1)
            sorted_logits, sorted_indices = torch.sort(dis)
        return center,sorted_indices[:int(graph.num_nodes()*rate)]

    def loss_fun(self, out, center):
        """
        This is a function used to calculate the error loss of the model.

        Parameters
        ----------
        out : tensor
            Output of the model.
        
        center : numpy.ndarray
            The center vector of all samples.

        model : tensor
            The model we trained.

        super_param : float
            A hyperparameter that takes values in [0, 1].

        device : str
            The number of times you want to train the model.

        Returns
        -------
        loss : tensor
            The loss of model output.
        """
        loss = torch.sqrt(torch.sum((out - center) ** 2,dim=-1))
        loss = torch.mean(loss) 
        return loss


class model_base(nn.Module):
    """
    This is the basic structure model of AAGNN.

    Parameters
    ----------
    in_feats : int
        The feature dimension of the input data.
    out_feats : int
        The dimension of output feature.

    Examples
    -------
    >>> self.model = model_base(in_feats, out_feats)
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.line = nn.Linear(in_feats, out_feats,bias=False)
        self.line2 = nn.Linear(out_feats, out_feats,bias=False)
        self.detgat = detGAT(out_feats)
        self.drop = nn.Dropout(0.2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0,0.5)
    
    def forward(self, graph, x):
        z = self.line(x)
        z = self.drop(z)
        z = F.relu(z)
        z = self.line2(z)
        z = self.drop(z)
        z = F.relu(z)
        h = self.detgat(graph,z)
        return F.relu(h)
    
class detGAT(nn.Module):
    def __init__(self,in_feats) :
        super().__init__()
        self.LeakyReLU = nn.LeakyReLU()
        self.a = nn.Parameter(torch.rand(2*in_feats,1))

    def message_func(self,edges):
        cat = torch.concat([edges.dst['h'],edges.src['h']],dim=1)
        msg = {}
        msg['cat'] = self.LeakyReLU(cat @ self.a)
        msg['val'] = edges.src['h']
        return msg 
    def forward(self,graph,feat):
        with graph.local_scope():
            graph.ndata['h'] = feat
            graph.apply_edges(self.message_func)
            graph.edata['w'] = dgl.nn.functional.edge_softmax(graph,graph.edata['cat'])
            graph.edata['val'] *= graph.edata['w']
            graph.update_all(fn.copy_e('val','val'),fn.sum('val','res'))
            return graph.ndata['h'] - graph.ndata['res']
