"""
This is dataset loading and processing program for SL-GAD
"""

import os
from os import path as osp
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

import torch
import dgl
from dgl.data import DGLDataset
from dgl.nn.pytorch import EdgeWeightNorm

import sys 
sys.path.append('.\\.\\')

from common.dataset import GraphNodeAnomalyDectionDataset
from common.sample import CoLASubGraphSampling, UniformNeighborSampling, SLGAD_SubGraphSampling
from datetime import datetime

# torch.set_default_tensor_type(torch.DoubleTensor)

def safe_add_self_loop(g):
    """
    Add the self loop in g
    Parameters
    ----------
    g : DGL.graph
        the graph to add self loop

    Returns
    -------
    newg : dgl.heterograph.DGLHeteroGraph
        the graph has been added self loop
    """
    newg = dgl.remove_self_loop(g)
    newg = dgl.add_self_loop(newg)
    return newg

class SL_GAD_DataSet(DGLDataset):
    """
    This is a class to generate subgraph of dataset

    Parameters
    ----------
    base_dataset_name : str, optional
        the name of dataset, defaulr Cora
    subgraph_size : int, optional
        the size of subgraph, default 4
    args : arg.parser, optional
        the extra parameter from arg.parser, default None
    g_data : DGL.graph, optional
        graph of dataset made manually, default None
    g_data : torch.Tensor
        anomaly label of dataset made manually, default None
    """
    def __init__(self, base_dataset_name='Cora', subgraphsize=4, args = None, g_data = None, y_data = None):
        super(SL_GAD_DataSet).__init__()
        # print(g_data.ndata['feat'].dtype)
        # exit()
        self.args = args
        self.dataset_name = base_dataset_name
        self.subgraphsize = subgraphsize
        self.oraldataset = GraphNodeAnomalyDectionDataset(name=self.dataset_name, g_data = g_data, y_data = y_data)
        self.dataset = self.oraldataset[0]
        # print(self.dataset)
        # exit()
        # print(self.dataset.ndata['feat'][:5, :5])
        # exit()
        self.SLGAD_subgraphsampler = SLGAD_SubGraphSampling(length=self.subgraphsize)
        self.paces = []
        # self.write_graph()

        self.normalize_feat()
        self.normalize_adj()
        self.random_walk_sampling()
        self.adj_matrix()

    def adj_matrix(self):
        """
        functions to store adjacency matrix
        
        Paramaters
        ----------
        None
        
        Returns
        -------
        None
        """
        # print(self.dataset)
        # print(self.dataset.edata['w'])
        # exit()
        edge_weight = self.dataset.edata['w']
        edges = self.dataset.edges()
        src = edges[0]
        dst = edges[1]
        num_nodes = self.dataset.number_of_nodes()
        # print(edge_weight)
        # print(src)
        # print(dst)
        adj_matrix = sp.coo_matrix((edge_weight, (src, dst)), shape = (num_nodes, num_nodes))
        # print(adj_matrix)
        adj_matrix = adj_matrix.todense()
        # print(adj_matrix)
        self.adj_matrix = adj_matrix
        # exit()

    def write_graph(self):
        print("write_graph")
        graph = self.dataset
        edges = graph.edges()
        file = open("graph.txt", "w")
        file.write("%d %d\n"%(graph.number_of_nodes(), graph.number_of_edges()))
        for i in range(graph.number_of_edges()):
            # print(edges[0][i].item(), edges[1][i].item())
            file.write("%d %d\n"%(edges[0][i].item(), edges[1][i].item()))
        attr_num = graph.ndata["feat"].shape[1]
        file.write("%d %d\n"%(graph.number_of_nodes(), attr_num))
        # attr_str = ' '.join(str(_) for _ in graph.ndata["feat"][0].tolist())

        # attr_str = ' '.join(map(str, graph.ndata["feat"][0].tolist()))
        # # print(graph.ndata["feat"][0].tolist())
        # file.write(attr_str)
        
        for i in range(graph.number_of_nodes()):
            attr_str = ' '.join(map(str, graph.ndata["feat"][i].tolist()))
            # print(graph.ndata["feat"][0].tolist())
            file.write(attr_str)
            file.write("\n")
        file.write("%d\n"%(graph.number_of_nodes()))
        label_str = ' '.join(map(str, graph.ndata["label"].tolist()))
        file.write(label_str)
        # print(label_str)
        file.write("\n")
        file.close()

        file = open("start_nodes.txt", "w")
        file.write("%d %d\n"%(graph.number_of_nodes(), self.args.subgraph_size))
        start_nodes = [_ for _ in range(graph.number_of_nodes())]
        start_nodes_str = ' '.join(map(str, start_nodes))
        file.write(start_nodes_str)
        file.write("\n")
        file.close()
        # print(graph)

        # exit()
    def normalize_feat(self):
        """
        functions to normalize the features of nodes in graph
        
        Paramaters
        ----------
        None
        
        Returns
        -------
        None
        """
        # print(self.dataset.ndata['feat'][:5, :5])
        # exit()
        self.dataset.ndata['feat'] = F.normalize(self.dataset.ndata['feat'], p=1, dim=1)
    
    def normalize_adj(self):
        """
        functions to normalize the edge weight in graph
        
        Paramaters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.sample_graph = self.dataset
        norm = EdgeWeightNorm(norm='both')
        # self.dataset = dgl.remove_self_loop(self.dataset)
        # print(self.dataset)
        # print(self.dataset.num_edges())
        # print(type(self.dataset))
        norm_edge_weight = norm(self.dataset, edge_weight=torch.ones(self.dataset.num_edges()))

        edges_tuple = self.dataset.edges()
        src = edges_tuple[0]
        dst = edges_tuple[1]
        in_degrees = self.dataset.in_degrees()
        out_degrees = self.dataset.out_degrees()

        src_out_degrees = out_degrees[src]
        dst_in_degrees = in_degrees[dst]
        self.dataset = dgl.add_self_loop(self.dataset)
        self_loop_weight = torch.ones(self.dataset.number_of_nodes())
        norm_edge_weight = torch.cat((norm_edge_weight, self_loop_weight), dim = 0)
        self.dataset.edata['w'] = norm_edge_weight
        print('sum of edge weight : ', torch.sum(norm_edge_weight))
        # print(type(norm_edge_weight))
        # print(norm_edge_weight.shape)
        # print(self.dataset.edata['w'][:20])
        # print(self.dataset.edges()[0][:20])
        # print(self.dataset.edges()[1][:20])
        # print(norm_edge_weight[-10:])
        # exit()
        # self.dataset.edata['w'] = norm_edge_weight
        # print(norm_edge_weight)

    def random_walk_sampling(self):
        """
        functions to get random walk from target nodes
        Paramaters
        ----------
        None

        Returns
        -------
        None
        """
        time_0 = datetime.now()
        # print(self.sample_graph)
        # print(torch.sum(self.dataset.edata['w']))
        self.paces_1 = self.SLGAD_subgraphsampler(self.sample_graph, list(range(self.sample_graph.num_nodes())))
        time_1 = datetime.now()
        self.paces_2 = self.SLGAD_subgraphsampler(self.sample_graph, list(range(self.sample_graph.num_nodes())))
        time_2 = datetime.now()
        self.paces_3 = self.SLGAD_subgraphsampler(self.sample_graph, list(range(self.sample_graph.num_nodes())))
        time_3 = datetime.now()

        # print(self.paces_1[:20])
        # exit()
        # self.paces_1 = torch.tensor(self.paces_1)
        # print(self.paces_1)
        # equal_node = self.paces_1 == self.paces_1[:, 0: 1].repeat(1, 4)
        # print(torch.sum(equal_node))
        # exit()
        # print(self.dataset)
        # print(torch.sum(self.dataset.edata['w']))
        # exit()
        # print(self.paces_1[:20])
        # print(self.paces_2[:20])
        # exit()
        # print(time_1 - time_0)
        # print(time_2 - time_1)
        # print(time_3 - time_2)
        
        pass

    def graph_transform(self, g):
        """
        functions to transfrom graph

        Paramaters
        ----------
        g : DGL.Graph
            the graph to transform

        Returns
        -------
        newg : DGL.Graph
            the graph after transform
        """
        newg = g
        # newg = safe_add_self_loop(g)
        # add virtual node as target node.
        # newg.add_nodes(1)
        # newg.ndata['feat'][-1] = newg.ndata['feat'][0]
        # newg = safe_add_self_loop(newg)
        # Anonymization
        # newg.ndata['feat'][0] = 0
        return newg
    
    def construct_graph(self, pace):
        """
        Functions that construct the ith subgraph

        Parameters
        ----------
        pace : list
            random walk, the set of node to construct graph

        Returns
        -------
        temp_graph : DGL.Graph
            the subgraph
        """
        slice_adj = self.adj_matrix[pace, :][:, pace]
        temp_graph = dgl.from_scipy(sp.coo_matrix(slice_adj), eweight_name = 'w')
        temp_graph.ndata['feat'] = self.dataset.ndata['feat'][pace]
        temp_graph.ndata['Raw_Feat'] = self.dataset.ndata['Raw_Feat'][pace]
        # print(self.dataset.ndata['feat'][pace])
        # print(temp_graph.ndata['feat'])
        # exit()
        return temp_graph

    def __getitem__(self, i):
        """
        Functions that get the ith subgraph set, including two positive subgraphs and one negative subgraph

        Parameters
        ----------
        i : int
            The index of subgraph.

        Returns
        -------
        pos_subgraph_1 : dgl.heterograph.DGLHeteroGraph
            the first positive subgraph of ith subgraph set
        pos_subgraph_2 : dgl.heterograph.DGLHeteroGraph
            the second positive subgraph of ith subgraph set
        neg_subgraph : dgl.heterograph.DGLHeteroGraph
            the negative subgraph of ith subgraph set
        """
        # self.paces_1[i] = [0, 633, 1862, 2582]
        # print(self.dataset)
        if False and i < 20:
            # self.paces_1[i] = [0, 479, 633, 655]
            # self.paces_1[1] = [1, 652, 2, 654]
            # self.paces_1[2] = [2, 906, 867, 962]
            # self.paces_1[3] = [3, 3, 2544, 3]
            # self.paces_1[4] = [4, 1761, 1016, 1256]
            print(self.paces_1[i])
        # ori_graph = self.dataset
        # temp_graph = dgl.node_subgraph(self.dataset, self.paces_15])
        # print(ori_graph.edges()[0][:10])
        # print(ori_graph.edges()[1][:10])
        
        # print(ori_graph.out_edges(self.paces_1[i][0]))
        # print(ori_graph.out_edges(self.paces_1[1]))
        # print(ori_graph.out_edges(self.paces_1[2]))
        # print(ori_graph.out_edges(self.paces_1[3]))
        
        # print(ori_graph.in_edges(self.paces_1[0]))
        # print(ori_graph.in_edges(self.paces_1[1]))
        # print(ori_graph.in_edges(self.paces_1[2]))
        # print(ori_graph.in_edges(self.paces_1[3]))

        # print(temp_graph.edges()[0][:20])
        # print(temp_graph.edges()[1][:20])
        # print(temp_graph.edata['w'][:20])

        # temp_graph = self.construct_graph(self.paces_1[i])
        # pos_subgraph_1 = self.graph_transform(dgl.node_subgraph(self.dataset, self.paces_1[i]))
        # pos_subgraph_2 = self.graph_transform(dgl.node_subgraph(self.dataset, self.paces_2[i]))
        pos_subgraph_1 = self.construct_graph(self.paces_1[i])
        pos_subgraph_2 = self.construct_graph(self.paces_2[i])
        
        neg_idx = np.random.randint(self.dataset.num_nodes()) 
        # while neg_idx == i or self.dataset.ndata["label"][neg_idx] == self.dataset.ndata["label"][i] or self.dataset.has_edges_between(neg_idx, i):
        while neg_idx == i:
            neg_idx = np.random.randint(self.dataset.num_nodes()) 
        # neg_subgraph = self.graph_transform(dgl.node_subgraph(self.dataset, self.paces_3[neg_idx]))
        
        neg_subgraph = self.construct_graph(self.paces_3[neg_idx])
        return pos_subgraph_1, pos_subgraph_2, neg_subgraph, i

    def __len__(self):
        """
        get the number of nodes of graph

        Parameters
        ----------
        None

        Returns
        -------
        num : int
            number of nodes of graph
        """
        return self.dataset.num_nodes()

    def process(self):
        """
        nonsense

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

if __name__ == '__main__':

    dataset = SL_GAD_DataSet()
    # print(dataset[0].edges())
    ans = []
    for i in range(100):
        dataset.random_walk_sampling()
        ans.append(dataset[502][1].ndata[dgl.NID].numpy().tolist())
    print(set([str(t) for t in ans]))
    # graph, label = dataset[0]
    # print(graph, label)
