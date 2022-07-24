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
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--restart', type=float, default=0.)
    parser.add_argument('--num_neighbors', type=int, default=-1)
    parser.add_argument('--embedding_dim', type=int, default=32)
    
    args = parser.parse_args()

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)
            
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
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
        },
        "fit":{
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "logdir": args.logdir,
            "num_epoch": args.num_epoch,
            "device": args.device,
            "batch_size": args.batch_size,
            "num_neighbors": args.num_neighbors,
            "max_len": args.max_len, 
            "restart": args.restart,
        },
        "predict":{
            "device": args.device,
            "batch_size": args.batch_size,
            "max_len": args.max_len, 
            "restart": args.restart,
        }
    }
    return final_args_dict


def random_walk_with_restart(g:dgl.DGLGraph, k=3, r=0.3, eps=1e-5):
    """Consistent with the description of "Network Preprocessing" in Section 4.1 of the paper.

    Parameters
    ----------
    g : dgl.DGLGraph
        graph data
    k : int, optional
        The maximum length of the truncated random walk, by default 3
    r : float, optional
        Probability of restart, by default 0.3
    eps : float, optional
        To avoid errors when the reciprocal of a node's out-degree is inf, by default 0.1

    Returns
    -------
    torch.Tensor
    """
    newg = deepcopy(g)
    # newg = newg.remove_self_loop().add_self_loop()
    # D^-1
    inv_degree = torch.pow(newg.out_degrees().float(), -1)
    inv_degree = torch.where(torch.isinf(inv_degree), torch.full_like(inv_degree, eps), inv_degree)
    inv_degree = torch.diag(inv_degree)
    # A
    adj = newg.adj().to_dense()
    mat = inv_degree @ adj 
    
    P_0 = torch.eye(newg.number_of_nodes()).float()
    X = torch.zeros_like(P_0) 
    P = P_0 
    for i in range(k): 
        P = r * P @ mat + (1 - r) * P_0 
        X += P 
    X /= k
    
    return X
    

def loss_func(x, x_hat, c, c_hat, h_a, h_s, hom_str, hom_attr, alphas, scale_factor, pretrain=False):
    """_summary_

    Parameters
    ----------
    x : torch.Tensor
        _description_
    x_hat : torch.Tensor
        _description_
    c : torch.Tensor
        _description_
    c_hat : torch.Tensor
        _description_
    h_a : torch.Tensor
        _description_
    h_s : torch.Tensor
        _description_
    hom_str : torch.Tensor
        _description_
    hom_attr : torch.Tensor
        _description_
    alphas : list
        _description_
    scale_factor : float
        _description_
    pretrain : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # closed form update rules
    # Eq.8 struct score
    ds = torch.norm(x-x_hat, dim=1)
    numerator = alphas[0] * ds + alphas[1] * hom_str
    os = numerator / torch.sum(numerator) * scale_factor
    # Eq.9 attr score
    da = torch.norm(c-c_hat, dim=1)
    numerator = alphas[2] * da + alphas[3] * hom_attr
    oa = numerator / torch.sum(numerator) * scale_factor
    # Eq.10 com score
    dc = torch.norm(h_s-h_a, dim=1)
    oc = dc / torch.sum(dc) * scale_factor
    
    # using Adam
    if pretrain is True:
        loss_prox_str = torch.mean(ds)          # Eq.2
        loss_hom_str = torch.mean(hom_str)      # Eq.3
        loss_prox_attr = torch.mean(da)         # Eq.4
        loss_hom_attr = torch.mean(hom_attr)    # Eq.5
        loss_com = torch.mean(dc)               # Eq.6
    else:
        loss_prox_str = torch.mean(torch.log(torch.pow(os, -1)) * ds)       # Eq.2
        loss_hom_str = torch.mean(torch.log(torch.pow(os, -1)) * hom_str)   # Eq.3
        loss_prox_attr = torch.mean(torch.log(torch.pow(oa, -1)) * da)      # Eq.4
        loss_hom_attr = torch.mean(torch.log(torch.pow(oa, -1)) * hom_attr) # Eq.5
        loss_com = torch.mean(torch.log(torch.pow(oc, -1)) * dc)            # Eq.6
        
    # sum = loss_prox_str + loss_hom_str + loss_prox_attr + loss_hom_attr + loss_com
    # print("a0={:.3f}".format((loss_prox_str/sum).item()), 
    #       "a1={:.3f}".format((loss_hom_str/sum).item()), 
    #       "a2={:.3f}".format((loss_prox_attr/sum).item()), 
    #       "a3={:.3f}".format((loss_hom_attr/sum).item()), 
    #       "a4={:.3f}".format((loss_com/sum).item()), 
    #       )
    # Eq.7
    loss = alphas[0] * loss_prox_str + alphas[1] * loss_hom_str + alphas[2] * loss_prox_attr + alphas[3] * loss_hom_attr + alphas[4] * loss_com
    
    score = (oa + os + oc) / 3
    return loss, score
  
    
def train_step(model, optimizer:torch.optim.Optimizer, g:dgl.DGLGraph, adj:torch.Tensor, batch_size:int, alphas:list, num_neighbors:int, device, pretrain=False):
    """_summary_

    Parameters
    ----------
    model : _type_
        _description_
    optimizer : torch.optim.Optimizer
        _description_
    g : dgl.DGLGraph
        _description_
    adj : torch.Tensor
        _description_
    batch_size : int
        _description_
    alphas : list
        _description_
    num_neighbors : int
        _description_
    device : _type_
        _description_
    pretrain : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # g = deepcopy(g)
    model.train()
    sampler = SubgraphNeighborSampler(num_neighbors)
    dataloader = dgl.dataloading.DataLoader(
        g, torch.arange(g.num_nodes()), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    epoch_loss = 0
    predict_score = torch.zeros(g.num_nodes())
    
    for sg in dataloader:
        feat = sg.ndata['feat']
        indices = sg.ndata['_ID']
        sub_adj = adj[indices]
        scale_factor = 1.0 * sg.num_nodes() / g.num_nodes()
        
        sg = sg.to(device)
        sub_adj = sub_adj.to(device)
        feat = feat.to(device)
        
        h_s, x_hat, h_a, c_hat, hom_str, hom_attr = model(sg, sub_adj, feat)
        loss, score = loss_func(sub_adj, x_hat, feat, c_hat, h_a, h_s, hom_str, hom_attr, alphas, scale_factor, pretrain=pretrain)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * sg.num_nodes()
        predict_score[indices] = score.detach().cpu()
        
    epoch_loss /= g.num_nodes()
        
    return predict_score, epoch_loss


def test_step(model, g, adj, batch_size, alphas, device):
    """_summary_

    Parameters
    ----------
    model : _type_
        _description_
    g : _type_
        _description_
    adj : _type_
        _description_
    batch_size : _type_
        _description_
    alphas : _type_
        _description_
    device : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # g = deepcopy(g)
    model.eval()
    sampler = SubgraphNeighborSampler()
    dataloader = dgl.dataloading.DataLoader(
        g, torch.arange(g.num_nodes()), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    predict_score = torch.zeros(g.num_nodes())
    
    for sg in tqdm(dataloader):
        feat = sg.ndata['feat']
        indices = sg.ndata['_ID']
        sub_adj = adj[indices]
        scale_factor = 1.0 * sg.num_nodes() / g.num_nodes()
        
        sg = sg.to(device)
        sub_adj = sub_adj.to(device)
        feat = feat.to(device)
        
        h_s, x_hat, h_a, c_hat, hom_str, hom_attr = model(sg, sub_adj, feat)
        _, score = loss_func(sub_adj, x_hat, feat, c_hat, h_a, h_s, hom_str, hom_attr, alphas, scale_factor)

        predict_score[indices] = score.detach().cpu()
        
    return predict_score.numpy()


class SubgraphNeighborSampler(dgl.dataloading.Sampler):
    """_summary_

    Parameters
    ----------
    num_neighbors : int, optional
        _description_, by default -1
    """
    def __init__(self, num_neighbors=-1):
        super().__init__()
        self.num_neighbors = num_neighbors

    def sample(self, g, indices):
        g = g.subgraph(indices)
        g = dgl.sampling.sample_neighbors(g, g.nodes(), self.num_neighbors)
        return g