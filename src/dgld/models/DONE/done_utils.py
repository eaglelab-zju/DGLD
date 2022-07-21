import shutil
import sys
import os
sys.path.append('../../')

import argparse
from tqdm import tqdm
from copy import deepcopy

import torch
import dgl


def get_parse():
    """
    get hyperparameter by parser from command line

    Returns
    -------
    final_args_dict : dictionary
        dict of args parser
    """
    parser = argparse.ArgumentParser(
        description='CONAD: Contrastive Attributed Network Anomaly Detection with Data Augmentation')

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logdir', type=str, default='tmp')
    parser.add_argument('--num_epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=0)
    
    args = parser.parse_args()

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    if args.lr is None:
        if args.dataset in ['Cora', 'Citeseer', 'Pubmed', 'Flickr']:
            args.lr = 1e-3
        elif args.dataset == 'ACM':
            args.lr = 5e-4
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3
        elif args.dataset == 'ogbn-arxiv':
            args.lr = 1e-3

    if args.num_epoch is None:
        if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog', 'Flickr', 'ACM']:
            args.num_epoch = 400
        else:
            args.num_epoch = 10
            
    in_feature_map = {
        "Cora":1433,
        "Citeseer":3703,
        "Pubmed":500,
        "BlogCatalog":8189,
        "Flickr":12047,
        "ACM":8337,
        "ogbn-arxiv":128,
    }
    num_nodes_map={
        "Cora":2708,
        "Citeseer":3327,
        "Pubmed":19717,
        "BlogCatalog":5196,
        "Flickr":7575,
        "ACM":16484,
        "ogbn-arxiv":169343,
    }
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model":{
            "feat_size": in_feature_map[args.dataset],
            "num_nodes": num_nodes_map[args.dataset],
        },
        "fit":{
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "logdir": args.logdir,
            "num_epoch": args.num_epoch,
            "device": args.device,
            "batch_size": args.batch_size,
        },
        "predict":{
            "device": args.device,
            "batch_size": args.batch_size,
        }
    }
    return final_args_dict


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
        P = r * P @ mat + (1 - r) * P_0 # BUG: fix
        X += P 
    X /= k
    
    newg = dgl.graph(X.nonzero(as_tuple=True))
    newg.edata[eweight_name] = X[newg.edges()]
    newg.ndata['feat'] = g.ndata['feat']
    return newg


class RWR(dgl.transforms.BaseTransform):
    def __init__(self, T=3, r=0.3):
        super(RWR, self).__init__()
        self.T = T
        self.r = r
        
    def __call__(self, g, eweight_name='w'):
        newg = deepcopy(g).remove_self_loop()
    
        adj = newg.adj().to_dense()
        inv_degree = torch.pow(newg.out_degrees().float(), -1)
        inv_degree = torch.where(torch.isinf(inv_degree), torch.full_like(inv_degree, 0.1), inv_degree)
        
        P_0 = torch.eye(newg.number_of_nodes()).float()
        mat = inv_degree @ adj
        X = P = P_0
        for i in range(self.T):
            P = self.r * P @ mat + (1 - self.r) * P_0 
            X += P 
        X /= self.T
        
        newg = dgl.graph(X.nonzero(as_tuple=True))
        newg.edata[eweight_name] = X[newg.edges()]
        return newg    
    

def loss_func(x, x_hat, c, c_hat, h_a, h_s, hom_str, hom_attr, alphas:dict, pretrain=False):
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
    
    # using Adam
    if pretrain is True:
        loss_prox_str = torch.mean(dx_norm)     # Eq.2
        loss_hom_str = torch.mean(hom_str)      # Eq.3
        loss_prox_attr = torch.mean(da_norm)    # Eq.4
        loss_hom_attr = torch.mean(hom_attr)    # Eq.5
        loss_com = torch.mean(dc_norm)          # Eq.6
    else:
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
  
    
def train_step(model, optimizer:torch.optim.Optimizer, graph:dgl.DGLGraph, adj:torch.Tensor,batch_size:int, alphas:dict, num_neighbors:int, pretrain=False):
    sampler = SubgraphNeighborSampler(num_neighbors)
    dataloader = dgl.dataloading.DataLoader(
        graph, torch.arange(graph.num_nodes()), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    epoch_loss = 0
    predict_score = torch.zeros(graph.num_nodes())
    
    # adj = graph.remove_self_loop().adj().to_dense()
    
    for sg in dataloader:
        feat = sg.ndata['feat']
        sub_adj = adj[sg.ndata['_ID']]
        
        h_s, x_hat, h_a, c_hat, hom_str, hom_attr = model(sg, sub_adj, feat)
        loss, score = loss_func(sub_adj, x_hat, feat, c_hat, h_a, h_s, hom_str, hom_attr, alphas, pretrain=pretrain)
        
        epoch_loss += loss * sg.num_nodes()
        predict_score[sg.ndata['_ID']] = score.cpu().detach()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_loss /= graph.num_nodes()
        
    return predict_score, epoch_loss


def test_step(model, graph, adj, batch_size, alphas):
    sampler = SubgraphNeighborSampler()
    dataloader = dgl.dataloading.DataLoader(
        graph, torch.arange(graph.num_nodes()), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    predict_score = torch.zeros(graph.num_nodes())
    
    # adj = graph.adj().to_dense()
    
    for sg in tqdm(dataloader):
        feat = sg.ndata['feat']
        sub_adj = adj[sg.ndata['_ID']]
        
        h_s, x_hat, h_a, c_hat, hom_str, hom_attr = model(sg, sub_adj, feat)
        _, score = loss_func(sub_adj, x_hat, feat, c_hat, h_a, h_s, hom_str, hom_attr, alphas)

        predict_score[sg.ndata['_ID']] = score.cpu().detach()
        
    return predict_score.numpy()


class SubgraphNeighborSampler(dgl.dataloading.Sampler):
    def __init__(self, num_neighbors=-1):
        super().__init__()
        self.num_neighbors = num_neighbors

    def sample(self, g, indices):
        g = g.subgraph(indices)
        g = dgl.sampling.sample_neighbors(g, g.nodes(), self.num_neighbors)
        return g