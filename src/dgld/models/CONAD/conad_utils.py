import shutil
import sys
import os
sys.path.append('../../')

import argparse
import numpy as np
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

    parser.add_argument('--dataset', type=str, default='Flickr')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logdir', type=str, default='tmp')
    parser.add_argument('--num_epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='balance parameter')
    parser.add_argument('--eta', type=float, default=0.7,
                        help='Attribute penalty balance parameter')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--contrast_type', type=str, default='siamese')
    parser.add_argument('--rate', type=float, default=0.2)
    parser.add_argument('--margin', type=float, default=0.5)
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
      
    if args.dataset == 'Cora': 
        args.rate = 0.1
        args.alpha = 0.85
        args.eta = 0.8
    elif args.dataset == 'Citeseer':
        pass
    elif args.dataset == 'Pubmed':
        args.lr = 0.003
        args.alpha = 0.95
        args.eta = 0.01
    elif args.dataset == 'BlogCatalog':
        args.alpha = 0.2
        args.eta = 0.2
    elif args.dataset == 'Flickr':
        pass
    elif args.dataset == 'ACM':
        args.lr = 0.003
        args.alpha = 0.1
        args.eta = 0.4 
        args.contrast_type = 'triplet'
    elif args.dataset == 'ogbn-arxiv':
        args.batch_size = 1024
        args.num_epoch = 50
            
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
            "feat_size": in_feature_map[args.dataset] if args.dataset in in_feature_map.keys() else None,
        },
        "fit":{
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "logdir": args.logdir,
            "num_epoch": args.num_epoch,
            "device": args.device,
            "eta": args.eta,
            "alpha": args.alpha,
            "rate": args.rate,
            "contrast_type": args.contrast_type,
            "margin": args.margin,
            "batch_size": args.batch_size,
        },
        "predict":{
            "device": args.device,
            "alpha": args.alpha,
            "batch_size": args.batch_size,
        }
    }
    return final_args_dict
  
    
def loss_func(a, a_hat, x, x_hat, alpha):
    """compute the loss function of the reconstructed graph

    Parameters
    ----------
    a : tensor.Torch
        The adjacency matrix of the original graph
    a_hat : tensor.Torch
        The adjacency matrix of the reconstructed graph    
    x : tensor.Torch
        feature matrix of the original graph
    x_hat : tensor.Torch
        feature matrix of the reconstructed graph
    alpha : float
        balance parameter

    Returns
    -------
    loss : torch.Tensor
        total loss
    struct_loss : torch.Tensor
        loss of reconstructed structure
    feat_loss : torch.Tensor
        loss of reconstructed features
    """
    # adjacency matrix reconstruction loss
    struct_norm = torch.linalg.norm(a-a_hat, dim=1)
    struct_loss = torch.mean(struct_norm)
    # feature matrix reconstruction loss
    feat_norm = torch.linalg.norm(x-x_hat, dim=1)
    feat_loss = torch.mean(feat_norm)
    # total reconstruction loss
    loss = (1-alpha) * struct_norm + alpha * feat_norm
    return loss, struct_loss, feat_loss


def train_step(model, optimizer, criterion, g_orig, g_aug, alpha, eta):
    """train model in one epoch

    Parameters
    ----------
    model : class
        CONAD model
    optimizer : optim.Adam
        optimizer to adjust model
    criterion : torch.nn.Functions
        functions to compute loss
    g_orig : dgl.DGLGraph
        original graph
    g_aug : dgl.DGLGraph
        augmented graph
    alpha : float
        balance parameter
    eta : float
        balance parameter

    Returns
    -------
    contrast_loss : torch.Tensor
        contrastive loss
    recon_loss : torch.Tensor
        total recontructed loss
    feat_loss : torch.Tensor
        recontructed feature loss    
    struct_loss : torch.Tensor
        recontructed structure loss    
    """
    model.train()
    feat_orig, feat_aug = g_orig.ndata['feat'], g_aug.ndata['feat']
    label = g_aug.ndata['label']
    # contrastive loss
    h_orig = model.embed(g_orig, feat_orig)
    h_aug = model.embed(g_aug, feat_aug)
    adj_orig = g_orig.adj().to_dense().to(label.device)
    contrast_loss = criterion(h_orig, h_aug, label, adj_orig)
    # recontruct loss
    a_hat, x_hat = model(g_orig, feat_orig)
    recon_loss, struct_loss, feat_loss = loss_func(adj_orig, a_hat, feat_orig, x_hat, alpha)
    # total loss
    loss = eta * contrast_loss + (1-eta) * recon_loss.mean()
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return contrast_loss, recon_loss, feat_loss, struct_loss


def train_step_batch(model, optimizer, criterion, g_orig, g_aug, alpha, eta, batch_size, device):
    """train model in one epoch for mini-batch graph training

    Parameters
    ----------
    model : class
        CONAD model
    optimizer : optim.Adam
        optimizer to adjust model
    criterion : torch.nn.Functions
        functions to compute loss
    g_orig : dgl.DGLGraph
        original graph
    g_aug : dgl.DGLGraph
        augmented graph
    alpha : float
        balance parameter
    eta : float
        balance parameter
    batch_size : int
        the size of training batch
    device : str
        device of computation

    Returns
    -------
    loss : float
        epoch loss
    """
    model.train()
    node_list = g_orig.nodes()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=3)
    dataloader_orig = dgl.dataloading.DataLoader(
        g_orig, node_list, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    dataloader_aug = dgl.dataloading.DataLoader(
        g_aug, node_list, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
            
    epoch_loss = 0
    
    # adj = g_orig.adj().to_dense()
    
    for (input_nodes_orig, output_nodes_orig, blocks_orig), (input_nodes_aug, output_nodes_aug, blocks_aug) in zip(dataloader_orig, dataloader_aug):
        blocks_orig = [b.to(device) for b in blocks_orig] 
        blocks_aug = [b.to(device) for b in blocks_aug] 
        
        input_feat_orig = blocks_orig[0].srcdata['feat']
        input_feat_aug = blocks_aug[0].srcdata['feat']
        
        label = blocks_aug[-1].dstdata['label']
        # contrastive loss
        h_orig = model.embed(blocks_orig, input_feat_orig)
        h_aug = model.embed(blocks_aug, input_feat_aug)
        adj_orig = dgl.node_subgraph(g_orig, output_nodes_orig).adj().to_dense().to(label.device)
        # adj_orig = adj[output_nodes_orig, output_nodes_orig].to(label.device)
        # print(h_orig.shape, h_aug.shape, adj_orig.shape, label.shape)
        output_nodes_num = blocks_orig[-1].number_of_dst_nodes()
        contrast_loss = criterion(h_orig[:output_nodes_num], h_aug[:output_nodes_num], label, adj_orig)
        # recontruct loss
        a_hat, x_hat = model(blocks_orig, input_feat_orig)
        output_feat_orig = blocks_orig[-1].dstdata['feat']
        recon_loss, struct_loss, feat_loss = loss_func(adj_orig, a_hat[:output_nodes_num, :output_nodes_num], output_feat_orig, x_hat[:output_nodes_num], alpha)
        # total loss
        loss = eta * contrast_loss + (1-eta) * recon_loss.mean()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_size
        
    return epoch_loss
    
    
def test_step(model, graph, alpha):
    """test model in one epoch

    Parameters
    ----------
    model : class
        CONAD model
    graph : dgl.DGLGraph
        graph dataset
    alpha : float
        balance parameter

    Returns
    -------
    score : numpy.ndarray
        anomaly scores of nodes
    """
    model.eval()
    feat = graph.ndata['feat']
    a_hat, x_hat = model(graph, feat)
    a_hat, x_hat = a_hat.cpu(), x_hat.cpu()

    adj_orig = graph.adj().to_dense().cpu()
    feat_orig = feat.detach().cpu()
    recon_loss, struct_loss, feat_loss = loss_func(adj_orig, a_hat, feat_orig, x_hat, alpha)
    score = recon_loss.detach().numpy()
    return score


def test_step_batch(model, graph, alpha, batch_size, device):
    """test model in one epoch for mini-batch graph training

    Parameters
    ----------
    model : class
        CONAD model
    graph : dgl.DGLGraph
        graph dataset
    alpha : float
        balance parameter
    batch_size : int
        the size of training batch
    device : str
        device of computation

    Returns
    -------
    score : numpy.ndarray
        anomaly scores of nodes
    """
    model.eval()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=3)
    dataloader = dgl.dataloading.DataLoader(
        graph, graph.nodes(), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    score = np.zeros(graph.num_nodes())
    
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        feat = blocks[0].srcdata['feat']
        a_hat, x_hat = model(blocks, feat)
        
        adj_orig = dgl.node_subgraph(graph, output_nodes).adj().to_dense().to(device)
        feat_orig = blocks[-1].dstdata['feat']
        output_nodes_num = blocks[-1].number_of_dst_nodes()
        recon_loss, struct_loss, feat_loss = loss_func(adj_orig, a_hat[:output_nodes_num, :output_nodes_num], feat_orig, x_hat[:output_nodes_num], alpha)
        score[output_nodes] = recon_loss.cpu().detach().numpy()
        
    return score