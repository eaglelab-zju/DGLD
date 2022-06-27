import os
from os import path as osp
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import dgl
from dgl.data import DGLDataset
from dgl.nn.pytorch import EdgeWeightNorm

from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.common.sample import CoLASubGraphSampling, UniformNeighborSampling

def safe_add_self_loop(g):
    newg = dgl.remove_self_loop(g)
    newg = dgl.add_self_loop(newg)
    return newg

class CoLADataSet(DGLDataset):
    """
    CoLA Dataset to generate subgraph for train and inference.
    
    Parameter:
    ----------
    g :  dgl.graph
    subgraphsize: int, optional
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
        self.dataset.ndata['feat'] = F.normalize(self.dataset.ndata['feat'], p=1, dim=1)
        norm = EdgeWeightNorm(norm='both')
        self.dataset = safe_add_self_loop(self.dataset)
        norm_edge_weight = norm(self.dataset, edge_weight=torch.ones(self.dataset.num_edges()))
        self.dataset.edata['w'] = norm_edge_weight
    def random_walk_sampling(self):
        self.paces = self.colasubgraphsampler(self.dataset, list(range(self.dataset.num_nodes())))

    def graph_transform(self, g):
        newg = g
        return newg

    def __getitem__(self, i):
        pos_subgraph = self.graph_transform(dgl.node_subgraph(self.dataset, self.paces[i]))
        neg_idx = np.random.randint(self.dataset.num_nodes()) 
        while neg_idx == i:
            neg_idx = np.random.randint(self.dataset.num_nodes()) 
        neg_subgraph = self.graph_transform(dgl.node_subgraph(self.dataset, self.paces[neg_idx]))
        return pos_subgraph, neg_subgraph

    def __len__(self):
        return self.dataset.num_nodes()

    def process(self):
        pass
