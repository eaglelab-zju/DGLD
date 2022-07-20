from random import sample
from re import sub
from statistics import mean
import torch

import dgl

import os
import wget
import numpy as np
from tqdm import tqdm
import csv
from copy import deepcopy



def random_walk_with_restart(g:dgl.DGLGraph, k=3, r=0.3, eweight_name='w', eps=0.1):
    """Consistent with the description of "Network Preprocessing" in Section 4.1 of the paper.

    Parameters
    ----------
    g : dgl.DGLGraph
        graph data
    k : int, optional
        The maximum length of the truncated random walk, by default 3
    r : float, optional
        Probability of restart, by default 0.3
    eweight_name : str, optional
        The name of the edge weight store, by default 'w'
    eps : float, optional
        To avoid errors when the reciprocal of a node's out-degree is inf, by default 0.1

    Returns
    -------
    dgl.DGLGraph
        
    Examples
    --------
    >>> g = dgl.rand_graph(4, 3)
    >>> newg = random_walk_with_restart(g)
    >>> print(g, newg)
    """
    newg = deepcopy(g)
    
    adj = newg.adj().to_dense()
    inv_degree = torch.pow(newg.out_degrees().float(), -1)
    inv_degree = torch.where(torch.isinf(inv_degree), torch.full_like(inv_degree, eps), inv_degree)
    
    P_0 = torch.eye(newg.number_of_nodes()).float()
    mat = inv_degree @ adj
    X = P = P_0
    for i in range(k):
        P = r * P @ mat + (1 - r) * P
        X += P 
    X /= k
    
    newg = dgl.graph(X.nonzero(as_tuple=True))
    newg.edata[eweight_name] = X[newg.edges()]
    return newg


def read_csv_file_as_numpy(filepath:str):
    """Read csv file into numpy format

    Parameters
    ----------
    filepath : str
        file path
    """
    with open(filepath, 'r') as fp:
        rd = csv.reader(fp)
        ret = []
        for row in tqdm(rd):
            ret.append([float(r) for r in row])
    return np.array(ret)


def load_paper_dataset(dataset):
    """Loading the paper dataset

    Parameters
    ----------
    dataset : str
        The dataset used in the experiments of the paper
    """
    if dataset == 'cora':
        adj = read_csv_file_as_numpy('test_data/cora/A_Final_permuted.csv')
        feat = read_csv_file_as_numpy('test_data/cora/C_Final_permuted.csv')
        label = read_csv_file_as_numpy('test_data/cora/labels_Final_permuted.csv')
        indices = read_csv_file_as_numpy('test_data/cora/permutation.csv')
    else:
        adj = read_csv_file_as_numpy('test_data/{}/struct.csv'.format(dataset))
        feat = read_csv_file_as_numpy('test_data/{}/content.csv'.format(dataset))
        label = read_csv_file_as_numpy('test_data/{}/label.csv'.format(dataset))
        indices = read_csv_file_as_numpy('test_data/{}/permutation.csv'.format(dataset))
    print(adj.shape, feat.shape, label.shape)
    print("#Nodes: {}".format(adj.shape[0]))
    print("#Edges: {}".format(adj.nonzero()[0].shape[0]))
    print("#Labels: ", np.unique(label).shape[0])
    print("#Attributes: {}".format(feat.shape[1]))
    
    graph = dgl.graph(adj.nonzero())
    graph.ndata['feat'] = torch.FloatTensor(feat)
    graph.ndata['label'] = torch.IntTensor(label)
    
    return graph, indices


def loss_func(x, x_hat, c, c_hat, h_a, h_s, hom_str, hom_attr, alphas:dict):
    # closed form update rules
    # Eq.8
    dx_norm = torch.norm(x-x_hat, dim=1)
    numerator = alphas[0] * dx_norm + alphas[1] * hom_str
    os = numerator / torch.sum(numerator)
    
    # Eq.9
    da_norm = torch.norm(c-c_hat, dim=1)
    numerator = alphas[2] * da_norm + alphas[3] * hom_attr
    oa = numerator / torch.sum(numerator)
    
    # Eq.10
    dc_norm = torch.norm(h_s-h_a, dim=1)
    oc = dc_norm / torch.sum(dc_norm)
    
    # Eq.2
    loss_prox_str = torch.mean(torch.log(torch.pow(os, -1)) * dx_norm)
    
    # Eq.3
    loss_hom_str = torch.mean(torch.log(torch.pow(os, -1)) * hom_str) 
    
    # Eq.4
    loss_prox_attr = torch.mean(torch.log(torch.pow(oa, -1)) * da_norm)
    
    # Eq.5
    loss_hom_attr = torch.mean(torch.log(torch.pow(oa, -1)) * hom_attr)
    
    # Eq.6
    loss_com = torch.mean(torch.log(torch.pow(oc, -1)) * dc_norm) 
    
    # Eq.7
    loss = alphas[0] * loss_prox_str + alphas[1] * loss_hom_str + alphas[2] * loss_prox_attr + alphas[3] * loss_hom_attr + alphas[4] * loss_com
    
    score = (oa + os + oc) / 3
    return loss, score

    
def train_step(model, optimizer:torch.optim.Optimizer, graph:dgl.DGLGraph, batch_size:int, alphas:dict, newg):
    sampler = SubgraphSampler()
    dataloader = dgl.dataloading.DataLoader(
        graph, torch.arange(graph.num_nodes()), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    epoch_loss = 0
    predict_score = torch.zeros(graph.num_nodes())
    
    adj = newg.adj().to_dense()
    
    
    for sg in dataloader:
        feat = sg.ndata['feat']
        sub_adj = adj[sg.ndata['_ID']]
        
        h_s, x_hat, h_a, c_hat, hom_str, hom_attr = model(sg, sub_adj, feat)
        loss, score = loss_func(sub_adj, x_hat, feat, c_hat, h_a, h_s, hom_str, hom_attr, alphas)
        
        epoch_loss += loss * sg.num_nodes()
        predict_score[sg.ndata['_ID']] = score.cpu().detach()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return predict_score, epoch_loss


def test_step(model, graph, batch_size, alphas):
    sampler = SubgraphSampler()
    dataloader = dgl.dataloading.DataLoader(
        graph, torch.arange(graph.num_nodes()), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    predict_score = torch.zeros(graph.num_nodes())
    
    adj = graph.adj().to_dense()
    
    for sg in tqdm(dataloader):
        feat = sg.ndata['feat']
        sub_adj = adj[sg.ndata['_ID']]
        
        h_s, x_hat, h_a, c_hat, hom_str, hom_attr = model(sg, sub_adj, feat)
        _, score = loss_func(sub_adj, x_hat, feat, c_hat, h_a, h_s, hom_str, hom_attr, alphas)

        predict_score[sg.ndata['_ID']] = score.cpu().detach()
        
    return predict_score.numpy()


class SubgraphSampler(dgl.dataloading.Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, g, indices):
        return g.subgraph(indices)
    

def recall_at_k(truth, score, k):
    ranking = np.argsort(-score)
    top_k = ranking[:k]
    top_k_label = truth[top_k]
    return top_k_label.sum() / truth.sum()