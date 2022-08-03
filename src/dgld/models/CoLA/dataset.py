import os
import sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
# print(current_dir)
sys.path.append(current_dir)
import torch.nn.functional as F
import numpy as np

import torch
import dgl
from dgl.data import DGLDataset
from dgl.nn.pytorch import EdgeWeightNorm

from utils.sample import CoLASubGraphSampling

def safe_add_self_loop(g):
    """
    Add the self loop in g
    Parameters
    ----------
    g : DGL.graph
        the graph to add self loop

    Returns
    -------
    newg : DGL.Graph
        the graph has been added self loop
    """
    newg = dgl.remove_self_loop(g)
    newg = dgl.add_self_loop(newg)
    return newg

class CoLADataSet(DGLDataset):
    """
    CoLA Dataset to generate subgraph for train and inference.
    
    Parameters
    ----------
    g :  dgl.graph
        graph to generate subgraph
    subgraphsize: int
        size of subgraph, default 4
    """
    def __init__(self, g, subgraphsize=4):
        super(CoLADataSet).__init__()
        self.subgraphsize = subgraphsize
        self.dataset = g
        self.colasubgraphsampler = CoLASubGraphSampling(length=self.subgraphsize)
        self.paces = []
        self.normalize_feat()
        self.random_walk_sampling()
    def normalize_feat(self):
        """
        functions to normalize the features of nodes in graph
        
        """
        self.dataset.ndata['feat'] = F.normalize(self.dataset.ndata['feat'], p=1, dim=1)
        norm = EdgeWeightNorm(norm='both')
        self.dataset = safe_add_self_loop(self.dataset)
        norm_edge_weight = norm(self.dataset, edge_weight=torch.ones(self.dataset.num_edges()))
        self.dataset.edata['w'] = norm_edge_weight
    def random_walk_sampling(self):
        """
        functions to get random walk from target nodes

        """
        self.paces = self.colasubgraphsampler(self.dataset, list(range(self.dataset.num_nodes())))

    def graph_transform(self, g):
        """
        functions to transfrom graph

        Parameters
        ----------
        g : DGL.Graph
            the graph to transform
        
        Returns
        -------
        newg : DGL.Graph
            the graph after transform
        """
        newg = g
        return newg

    def __getitem__(self, i):
        """
        Functions that get the ith subgraph set, including two positive subgraphs and one negative subgraph

        Parameters
        ----------
        i : int
            The index of subgraph.

        Returns
        -------
        pos_subgraph_1 : DGL.Graph
            the first positive subgraph of ith subgraph set
        pos_subgraph_2 : DGL.Graph
            the second positive subgraph of ith subgraph set
        neg_subgraph : DGL.Graph
            the negative subgraph of ith subgraph set
        """
        pos_subgraph = self.graph_transform(dgl.node_subgraph(self.dataset, self.paces[i]))
        neg_idx = np.random.randint(self.dataset.num_nodes()) 
        while neg_idx == i:
            neg_idx = np.random.randint(self.dataset.num_nodes()) 
        neg_subgraph = self.graph_transform(dgl.node_subgraph(self.dataset, self.paces[neg_idx]))
        return pos_subgraph, neg_subgraph

    def __len__(self):
        """
        get the number of nodes of graph

        Returns
        -------
        num : int
            number of nodes of graph
        """
        return self.dataset.num_nodes()

    def process(self):
        """
        nonsense

        """
        pass
