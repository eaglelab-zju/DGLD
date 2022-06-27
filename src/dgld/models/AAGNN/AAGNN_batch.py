"""
This is a model for large graph training based on the AAGNN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import euclidean
import scipy.sparse as spp
from tqdm import tqdm
from torch.autograd import Variable # torch 中 Variable 模块
from torch.utils.tensorboard import SummaryWriter

class AAGNN_batch(nn.Module):
    """
    This is a model for large graph training based on the AAGNN model.

    Parameters
    ----------
    in_feats : int
        The feature dimension of the input data.
    out_feats : int
        The dimension of output feature.

    Examples
    -------
    >>> from DGLD.AAGNN import AAGNN_batch
    >>> model = AAGNN_batch(in_feats=in_feats, out_feats=300)
    >>> model.fit(graph, num_epoch=30, device='cuda:0', subgraph_size=32)
    """
    def __init__(self, feat_size, out_dim=300):
        super().__init__()
        self.model = model_base(feat_size, out_dim)
        self.out_feats = out_dim
    
    def fit(self, graph, num_epoch=100, device='cpu', lr=0.0001, logdir='tmp', subgraph_size=4096):
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
        
        logdir: str
            The storage address of the training log.

        subgraph_size : int
            The size of training subgraph.

        Returns
        -------
        None
        """
        
        print('-'*40, 'training', '-'*40)
        print(graph)
        features = graph.ndata['feat']
        if device != 'cpu':
            device = 'cuda:' + device
        print('device=',device)
        model = self.model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        
        edge_dic = {}
        us = graph.edges()[0].numpy()
        vs = graph.edges()[1].numpy()
        for u,v in zip(us, vs):
            if v not in edge_dic.keys():
                edge_dic[v] = []
            edge_dic[v].append(u)
        
        node_ids = model.get_normal_nodes(features, 0.5, device)
        writer = SummaryWriter(log_dir=logdir)
        model.train()

        for epoch in range(num_epoch):
            center = self.cal_center(graph, model, device, subgraph_size, edge_dic)

            for index in range(0, len(node_ids), subgraph_size):
                L = index
                R = index + subgraph_size
                subgraph_node_ids = node_ids[L: R]
                adj_matrix, eye_matrix, subgraph_feats, node_mask = self.Graph_batch_sample(graph, subgraph_node_ids, device, edge_dic)

                out = model(adj_matrix, eye_matrix, subgraph_feats, node_mask)
                loss = self.loss_fun(out, center, model, 0.0001, device)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print("Epoch:", '%04d' % (epoch), "%04d/%04d"%(index, len(node_ids))," train_loss=", "{:.10f}".format(loss.item(
                )))

            
            writer.flush()

    def predict(self, graph, device='cpu', subgraph_size=4096):
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

        edge_dic = {}
        us = graph.edges()[0].numpy()
        vs = graph.edges()[1].numpy()
        for u,v in zip(us, vs):
            if v not in edge_dic.keys():
                edge_dic[v] = []
            edge_dic[v].append(u)

        score = []
        node_ids = np.arange(graph.ndata['feat'].shape[0])
        if device != 'cpu':
            device = 'cuda:' + device
            
        model = self.model.to(device)
        for index in range(0, len(node_ids), subgraph_size):
                L = index
                R = index + subgraph_size
                subgraph_node_ids = node_ids[L: R]
                adj_matrix, eye_matrix, subgraph_feats, node_mask = self.Graph_batch_sample(graph, subgraph_node_ids, device, edge_dic)

                out = model(adj_matrix, eye_matrix, subgraph_feats, node_mask)

                predict_score = model.anomaly_score(out)

                score += list(predict_score)
        
        return np.array(score)


    def Graph_batch_sample(self, graph, node_ids, device, edge_dic):
        """
        Here is a function for sampling the large graph data.

        Parameters
        ----------
        graph : dgl
            The graph data you input.
        
        node_ids : numpy
            The id of the target node you want to sample.

        device : str
            The number of times you want to train the model.
        
        edge_dic: dict
            The input node-to-node relationship dictionary.

        Returns
        -------
        adj_matrix : numpy.ndarray
            This is an adjacency matrix of sampled subgraphs.

        eye_matrix : numpy.ndarray
            This is an adjacency matrix of sampled subgraphs.

        subgraph_feats : numpy.ndarray 
            Node feature vector of the subgraph.

        node_mask : numpy.ndarray
            This is a vector marking which nodes are the target nodes we need.

        """

        feats = graph.ndata['feat']
        sample_u = []
        sample_v = []
        subgraph_feats = []

        for v in node_ids:
            for u in edge_dic[v]:
                
                sample_u.append(u)
                sample_v.append(v)
        sample_nodes = list(set(sample_u + sample_v))


        adj_matrix = np.zeros((len(sample_nodes), len(sample_nodes)))
        node_id_dic = {}

        for i in range(len(sample_nodes)):
            node_id_dic[sample_nodes[i]] = i
            subgraph_feats.append(feats[sample_nodes[i]].cpu().data.numpy())

        for u, v in zip(sample_u, sample_v):
            adj_matrix[node_id_dic[u]][node_id_dic[v]] = 1
            adj_matrix[node_id_dic[u]][node_id_dic[u]] = 1
            adj_matrix[node_id_dic[v]][node_id_dic[v]] = 1

        eye_matrix = torch.eye(len(sample_nodes))
        node_mask = []
        for id in node_ids:
            node_mask.append(node_id_dic[id])
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
        eye_matrix = torch.tensor(eye_matrix, dtype=torch.float32).to(device)
        subgraph_feats = torch.tensor(np.array(subgraph_feats), dtype=torch.float32).to(device)
        node_mask = torch.tensor(np.array(node_mask), dtype=torch.long).to(device)

        return adj_matrix, eye_matrix, subgraph_feats, node_mask

    def cal_center(self, graph, model, device, subgraph_size, edge_dic):
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

        center = torch.zeros(self.out_feats).to(device)

        with torch.no_grad():
            features = graph.ndata['feat']
            node_ids = np.arange(features.shape[0])
            for index in range(0, features.shape[0], subgraph_size):
                L = index
                R = index + subgraph_size
                subgraph_node_ids = node_ids[L: R]
                adj_matrix, eye_matrix, subgraph_feats, node_mask = self.Graph_batch_sample(graph, subgraph_node_ids, device, edge_dic)
                out = model(adj_matrix, eye_matrix, subgraph_feats, node_mask)

                center += torch.sum(out, dim=0)

            center = center/features.shape[0]

        return center

    def loss_fun(self, out, center, model, super_param, device):
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

        loss_matrix = torch.sum((out - center) * (out - center), dim=1)
        loss = torch.mean(loss_matrix, dim=0)
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        return loss + (super_param * l2_reg/2)


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
        self.line = nn.Linear(in_feats, out_feats)
        self.a_1 = nn.Parameter(torch.rand(1, out_feats))
        self.a_2 = nn.Parameter(torch.rand(1, out_feats))
        self.LeakyReLU = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, adj_matrix, eye_matrix, subgraph_feats, node_mask):
        """
        This is a function used to calculate the forward propagation of the model.

        Parameters
        ----------
        adj_matrix : tensor
            This is an adjacency matrix of sampled subgraphs.

        eye_matrix : tensor
            This is an adjacency matrix of sampled subgraphs.

        subgraph_feats : tensor
            Node feature vector of the subgraph.

        node_mask : tensor
            This is a vector marking which nodes are the target nodes we need.


        Returns
        -------
        h : tensor
            Results of model forward propagation calculations.
        """

        z = self.line(subgraph_feats)
        zi = torch.sum(self.a_1 * z, dim=1).reshape(-1, 1)
        zj = torch.sum(self.a_2 * z, dim=1).reshape(-1, 1)

        attention_A = adj_matrix * zi
        attention_B = adj_matrix * (eye_matrix * zj)
        attention_matrix = self.LeakyReLU(attention_A + attention_B)
        attention_matrix = self.softmax(attention_matrix)

        h = z - torch.mm(attention_matrix, z)
        return F.relu(h[node_mask])
    
    
    def get_normal_nodes(self, node_feats, p, device):
        """
        Here is a function that computes normal nodes.

        Parameters
        ----------
        node_feats : tensor
            All node feature vectors of graph data.
        
        p : float
            All node feature vectors of graph data.

        device : str
            The number of times you want to train the model.

        Returns
        -------
        node_ids : numpy.ndarray
            The ID of the positive sample node.
        """
        
        with torch.no_grad():
            node_feats = node_feats.to(device)
            z = self.line(node_feats)
            c = torch.mean(z, dim=0)
            dis = torch.sum((z - c) * (z - c), dim=1)
            best_min_dis = list(dis.cpu().data.numpy())
            best_min_dis.sort()
            threshold = best_min_dis[int(len(best_min_dis) * p)]

            node_ids = []
            for node_id in range(node_feats.shape[0]):
                if dis[node_id] <= threshold:
                    node_ids.append(node_id)

            return node_ids

    def anomaly_score(self, out):
        """
        Here is a function that calculates the anomaly score.

        Parameters
        ----------
        out : tensor
            Node vector representation output after model training.

        Returns
        -------
        score : numpy.ndarray
            Anomaly Score of Nodes.
        """

        s = torch.sum(out * out, dim=1)
        return s.cpu().data.numpy()