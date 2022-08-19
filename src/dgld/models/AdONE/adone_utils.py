from email.policy import default
import shutil
import sys
import os
sys.path.append('../../')

import argparse
from tqdm import tqdm
from copy import deepcopy

import random
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_wl
import dgl

from dgld.utils.common import loadargs_from_json

def set_subargs(parser):
    parser.add_argument('--num_epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--disc_update_times', type=int, default=1, help="number of discriminator updates")
    parser.add_argument('--gen_update_times', type=int, default=5, help="number of generator updates")
    parser.add_argument('--lr_all', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_disc', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_gen', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help="weight decay (L2 penalty)")
    parser.add_argument('--dropout', type=float, default=0., help="rate of dropout")
    parser.add_argument('--batch_size', type=int, default=0, help="size of training batch")
    parser.add_argument('--max_len', type=int, default=0, help="maximum length of the truncated random walk")
    parser.add_argument('--restart', type=float, default=0., help="probability of restart")
    parser.add_argument('--num_neighbors', type=int, default=-1, help="number of sampling neighbors")
    parser.add_argument('--embedding_dim', type=int, default=32, help="dimension of embedding")
    parser.add_argument('--verbose', type=lambda x: x.lower() == 'true', default=True, help="verbose or not")
    

def get_subargs(args): 
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model":{
            "feat_size": args.feat_dim,
            "num_nodes": args.num_nodes,
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
        },
        "fit":{
            "lr_all": args.lr_all,
            "lr_disc": args.lr_disc,
            "lr_gen": args.lr_gen,
            "weight_decay": args.weight_decay,
            "num_epoch": args.num_epoch,
            "disc_update_times": args.disc_update_times,
            "gen_update_times": args.gen_update_times,
            "device": args.device,
            "batch_size": args.batch_size,
            "num_neighbors": args.num_neighbors,
            "max_len": args.max_len, 
            "restart": args.restart,
            "verbose": args.verbose,
        },
        "predict":{
            "device": args.device,
            "batch_size": args.batch_size,
            "max_len": args.max_len, 
            "restart": args.restart,
        }
    }
    return final_args_dict, args


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
    

def loss_func(x, x_hat, c, c_hat, hom_str, hom_attr, dis_a, dis_s, betas, scale_factor, pretrain=False, train_disc=False, train_gen=False):
    """loss function

    Parameters
    ----------
    x : torch.Tensor
        adjacency matrix of the original graph
    x_hat : torch.Tensor
        adjacency matrix of the reconstructed graph
    c : torch.Tensor
        attribute matrix of the original graph
    c_hat : torch.Tensor
        attribute matrix of the reconstructed graph
    h_a : torch.Tensor
        embedding of attribute autoencoders
    h_s : torch.Tensor
        embedding of structure autoencoders
    hom_str : torch.Tensor
        intermediate value of homogeneity loss of structure autoencoder
    hom_attr : torch.Tensor
        intermediate value of homogeneity loss of attribute autoencoder
    dis_a : torch.Tensor
        discriminator output for attribute embeddings
    dis_s : torch.Tensor
        discriminator output for structure embeddings
    betas : list
        balance parameters
    scale_factor : float
        scale factor
    pretrain : bool, optional
        whether to pre-train, by default False

    Returns
    -------
    loss : torch.Tensor
        loss value
    score : torch.Tensor
        outlier score
    """
    # closed form update rules
    # Eq.8 struct score
    ds = torch.norm(x-x_hat, dim=1)
    numerator = betas[0] * ds + betas[1] * hom_str
    os = numerator / torch.sum(numerator) * scale_factor
    # Eq.9 attr score
    da = torch.norm(c-c_hat, dim=1)
    numerator = betas[2] * da + betas[3] * hom_attr
    oa = numerator / torch.sum(numerator) * scale_factor
    
    # using Adam
    if pretrain is True:
        loss_prox_str = torch.mean(ds)          # Eq.2
        loss_hom_str = torch.mean(hom_str)      # Eq.3
        loss_prox_attr = torch.mean(da)         # Eq.4
        loss_hom_attr = torch.mean(hom_attr)    # Eq.5
        loss_com = 0.
    else:
        loss_prox_str = torch.mean(torch.log(torch.pow(os, -1)) * ds) # Eq.2
        loss_hom_str = torch.mean(torch.log(torch.pow(os, -1)) * hom_str) # Eq.3
        loss_prox_attr = torch.mean(torch.log(torch.pow(oa, -1)) * da) # Eq.4
        loss_hom_attr = torch.mean(torch.log(torch.pow(oa, -1)) * hom_attr) # Eq.5
        loss_com = 0.
        
    if pretrain is False and train_disc is True:
        dc = (bce_wl(dis_a, torch.zeros_like(dis_a), reduction='none') + bce_wl(dis_s, torch.ones_like(dis_s), reduction='none')).squeeze()
        loss_disc = torch.mean(dc)
        return loss_disc
    
    dc = (bce_wl(dis_a, torch.ones_like(dis_a), reduction='none') + bce_wl(dis_s, torch.zeros_like(dis_s), reduction='none')).squeeze()
    oc = dc / torch.sum(dc) * scale_factor
    
    if pretrain is False and train_gen is True:
        loss_gen = torch.mean(torch.log(torch.pow(oc, -1)) * dc)
        return loss_gen
        
    # Eq.7
    loss_all = betas[0] * loss_prox_str + betas[1] * loss_hom_str \
        + betas[2] * loss_prox_attr + betas[3] * loss_hom_attr 
    
    score = (oa + os + oc) / 3
    return loss_all, score
  
    
def train_step(model, optimizer:torch.optim.Optimizer, g:dgl.DGLGraph, adj:torch.Tensor, batch_size:int, betas:list, num_neighbors:int, disc_update_times:int, gen_update_times:int, device:str, pretrain=False, verbose=False):
    """train model in one epoch

    Parameters
    ----------
    model : class
        AdONE base model
    optimizer : torch.optim.Optimizer
        optimizer to adjust model
    g : dgl.DGLGraph
        original graph
    adj : torch.Tensor
        adjacency matrix
    batch_size : int
        the size of training batch
    betas : list
        balance parameters
    num_neighbors : int
        number of sampling neighbors
    disc_update_times : int
        number of rounds of discriminator updates in an epoch
    gen_update_times : int
        number of rounds of auto-encoder updates in an epoch
    device : str
        device of computation
    pretrain : bool, optional
        whether to pre-train, by default False

    Returns
    -------
    predict_score : numpy.ndarray
        outlier score
    epoch_loss : torch.Tensor
        loss value for epoch
    """
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
    
    optim_all, optim_disc, optim_gen = optimizer
    
    for sg in dataloader:
        feat = sg.ndata['feat']
        indices = sg.ndata['_ID']
        sub_adj = adj[indices]
        scale_factor = 1.0 * sg.num_nodes() / g.num_nodes()
        
        sg = sg.to(device)
        sub_adj = sub_adj.to(device)
        feat = feat.to(device)
        
        # disc
        if pretrain is False:
            for i in range(disc_update_times):
                
                h_s, x_hat, h_a, c_hat, hom_str, hom_attr, dis_a, dis_s = model(sg, sub_adj, feat)
                loss_disc = loss_func(sub_adj, x_hat, feat, c_hat, hom_str, hom_attr, dis_a, dis_s, betas, scale_factor, pretrain=pretrain, train_disc=True)
                
                if verbose:
                    print("training disc...")
                    print('disc attr:\t', 'mean={:.4f}'.format(torch.mean(torch.sigmoid(dis_a)).item()), 
                        ', max={:.4f}'.format(torch.max(torch.sigmoid(dis_a)).item()), 
                        ', min={:.4f}'.format(torch.min(torch.sigmoid(dis_a)).item()))
                    print('disc struct:\t', 'mean={:.4f}'.format(torch.mean(torch.sigmoid(dis_s)).item()), 
                        ', max={:.4f}'.format(torch.max(torch.sigmoid(dis_s)).item()), 
                        ', min={:.4f}'.format(torch.min(torch.sigmoid(dis_s)).item()))
                                
                optim_disc.zero_grad()
                loss_disc.backward()
                optim_disc.step()
                
            for i in range(gen_update_times):
                
                h_s, x_hat, h_a, c_hat, hom_str, hom_attr, dis_a, dis_s = model(sg, sub_adj, feat)
                loss_gen = loss_func(sub_adj, x_hat, feat, c_hat, hom_str, hom_attr, dis_a, dis_s, betas, scale_factor, pretrain=pretrain, train_gen=True)
                
                if verbose:
                    print("training gen...")
                    print('gen attr:\t', 'mean={:.4f}'.format(torch.mean(torch.sigmoid(dis_a)).item()), 
                        ', max={:.4f}'.format(torch.max(torch.sigmoid(dis_a)).item()), 
                        ', min={:.4f}'.format(torch.min(torch.sigmoid(dis_a)).item()))
                    print('gen struct:\t', 'mean={:.4f}'.format(torch.mean(torch.sigmoid(dis_s)).item()), 
                        ', max={:.4f}'.format(torch.max(torch.sigmoid(dis_s)).item()), 
                        ', min={:.4f}'.format(torch.min(torch.sigmoid(dis_s)).item()))
                
                optim_gen.zero_grad()
                loss_gen.backward()
                optim_gen.step()

        h_s, x_hat, h_a, c_hat, hom_str, hom_attr, dis_a, dis_s = model(sg, sub_adj, feat)
        loss, score = loss_func(sub_adj, x_hat, feat, c_hat, hom_str, hom_attr, dis_a, dis_s, betas, scale_factor, pretrain=pretrain)
        
        optim_gen.zero_grad()
        loss.backward()
        optim_gen.step()
        
        epoch_loss += loss.item() * sg.num_nodes()
        predict_score[indices] = score.detach().cpu()
        
    epoch_loss /= g.num_nodes()
        
    return predict_score, epoch_loss


def test_step(model, g, adj, batch_size, betas, device):
    """test model in one epoch

    Parameters
    ----------
    model : nn.Module
        AdONE base model 
    g : dgl.DGLGraph
        graph data
    adj : torch.Tensor
        adjacency matrix
    batch_size : int
        the size of training batch
    betas : list
        balance parameters
    device : str
        device of computation

    Returns
    -------
    numpy.ndarray
    """
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
        
        h_s, x_hat, h_a, c_hat, hom_str, hom_attr, dis_a, dis_s = model(sg, sub_adj, feat)
        _, score = loss_func(sub_adj, x_hat, feat, c_hat, hom_str, hom_attr, dis_a, dis_s, betas, scale_factor)

        predict_score[indices] = score.detach().cpu()
        
    return predict_score.numpy()


class SubgraphNeighborSampler(dgl.dataloading.Sampler):
    """the neighbor sampler of the subgraph

    Parameters
    ----------
    num_neighbors : int, optional
        number of sampling neighbors, by default -1
    """
    def __init__(self, num_neighbors=-1):
        super(SubgraphNeighborSampler, self).__init__()
        self.num_neighbors = num_neighbors

    def sample(self, g, indices):
        g = g.subgraph(indices)
        g = dgl.sampling.sample_neighbors(g, g.nodes(), self.num_neighbors)
        return g