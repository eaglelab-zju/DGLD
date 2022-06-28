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
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from DGLD.common.dataset import split_auc

class AAGNN(nn.Module):
    """
    This is the model of AAGNN.

    Parameters
    ----------
    in_feats : int
        The feature dimension of the input data.
    out_feats : int
        The dimension of output feature.

    Examples
    -------
    >>> from DGLD.AAGNN import AAGNN
    >>> model = AAGNN(in_feats=in_feats, out_feats=300)
    >>> model.fit(graph, num_epoch=30, device='cuda:0', subgraph_size=32)
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.model = model_base(in_feats, out_feats)
    
    def fit(self, graph, num_epoch=100, device='cpu', lr=0.0001, logdir='tmp'):
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

        Returns
        -------
        None
        """

        print('-'*40, 'training', '-'*40)
        print(graph)
        features = graph.ndata['feat']
        features = features.to(device)
        model = self.model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        mask = model.mask_label(features, 0.5)
        writer = SummaryWriter(log_dir=logdir)
        
        num_nodes = len(graph.nodes().numpy())
        adj_matrix = np.zeros((num_nodes, num_nodes))

        us = graph.edges()[0].numpy()
        vs = graph.edges()[1].numpy()
        for u, v in zip(us, vs):
            adj_matrix[u][v] = 1.0
            adj_matrix[u][u] = 1.0
            adj_matrix[v][v] = 1.0

        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
        eye_matrix = torch.eye(num_nodes).to(device)

        for epoch in range(num_epoch):
            model.train()
            out = model(features, adj_matrix, eye_matrix)

            loss = model.loss_fun(out, mask, model, 0.0001, device)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.10f}".format(loss.item(
            )))
            writer.add_scalars(
                "loss",
                {"loss": loss},
                epoch,
            )

            writer.flush()

    def predict(self, graph, device='cpu'):
        """
        This is a function that loads a trained model for predicting graph data.

        Parameters
        ----------
        graph : dgl
            The graph data you input.
        
        device : str
            The number of times you want to train the model.

        Returns
        -------
        score : numpy
            A vector of decimals representing the anomaly score for each node.
        """

        print('-'*40, 'predicting', '-'*40)

        features = graph.ndata['feat']
        print(graph)

        features = features.to(device)
        model = self.model.to(device)

        num_nodes = len(graph.nodes().numpy())
        adj_matrix = np.zeros((num_nodes, num_nodes))

        us = graph.edges()[0].numpy()
        vs = graph.edges()[1].numpy()
        for u, v in zip(us, vs):
            adj_matrix[u][v] = 1.0
            adj_matrix[u][u] = 1.0
            adj_matrix[v][v] = 1.0

        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
        eye_matrix = torch.eye(num_nodes).to(device)

        out = model(features, adj_matrix, eye_matrix)
        predict_score = model.anomaly_score(out)
        return predict_score





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
    
    def forward(self, inputs, adj_matrix, eye_matrix):
        """
        This is a function used to calculate the forward propagation of the model.

        Parameters
        ----------
        inputs : tensor
            Input node feature vector.

        adj_matrix : tensor
            This is an adjacency matrix of sampled subgraphs.

        eye_matrix : tensor
            This is an adjacency matrix of sampled subgraphs.

        Returns
        -------
        h : tensor
            Results of model forward propagation calculations.
        """

        z = self.line(inputs)
        zi = torch.sum(self.a_1 * z, dim=1).reshape(-1, 1)
        zj = torch.sum(self.a_2 * z, dim=1).reshape(-1, 1)
        attention_A = adj_matrix * zi
        attention_B = adj_matrix * (eye_matrix * zj)
        attention_matrix = self.LeakyReLU(attention_A + attention_B)
        attention_matrix = self.softmax(attention_matrix)

        #print(np.max(attention_matrix.cpu().data.numpy()))
        h = z - torch.mm(attention_matrix, z)
        return F.relu(h)
    
    def mask_label(self, inputs, p):
        """
        Here is a function that computes normal nodes.

        Parameters
        ----------
        inputs : tensor
            All node feature vectors of graph data.
        
        p : float
            All node feature vectors of graph data.

        Returns
        -------
        node_ids : numpy.ndarray
            The ID of the positive sample node.
        """
        with torch.no_grad():
            z = self.line(inputs)
            c = torch.mean(z, dim=0)
            dis = torch.sum((z - c) * (z - c), dim=1)
            best_min_dis = list(dis.cpu().data.numpy())
            best_min_dis.sort()
            threshold = best_min_dis[int(len(best_min_dis) * p)]
            mask = (dis <= threshold)
            return mask
            
    def loss_fun(self, out, mask, model, super_param, device):
        """
        This is a function used to calculate the error loss of the model.

        Parameters
        ----------
        out : tensor
            Output of the model.
        
        mask : tensor
            This is a vector marking which nodes are positive samples.

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

        c = torch.mean(out, dim=0)

        loss_matrix = torch.sum((out - c) * (out - c), dim=1)[mask]

        loss = torch.mean(loss_matrix, dim=0)

        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        return loss + (super_param * l2_reg/2)
        

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
    
