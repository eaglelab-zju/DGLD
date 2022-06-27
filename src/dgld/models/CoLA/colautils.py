import argparse
from tqdm import tqdm
import numpy as np
import torch

import shutil
import sys
import os

from DGLD.utils.print import cprint, lcprint


def loss_fun_BPR(pos_scores, neg_scores, criterion, device):
    batch_size = pos_scores.shape[0]
    labels = torch.ones(batch_size).to(device)
    return criterion(pos_scores-neg_scores, labels)


def loss_fun_BCE(pos_scores, neg_scores, criterion, device):
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    batch_size = pos_scores.shape[0]
    pos_label = torch.ones(batch_size).to(device)
    neg_label = torch.zeros(batch_size).to(device)
    labels = torch.cat([pos_label, neg_label], dim=0)
    return criterion(scores, labels)


loss_fun = loss_fun_BCE


def get_parse():
    parser = argparse.ArgumentParser(
        description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
    # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=4)
    parser.add_argument('--auc_test_rounds', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--logdir', type=str, default='tmp')
    parser.add_argument('--global_adg', type=bool, default=True)
    args = parser.parse_args()

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)
    else:
        os.makedirs(args.logdir)

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

    if args.auc_test_rounds is None:
        if args.dataset != 'ogbn-arxiv':
            args.auc_test_rounds = 256
        else:
            args.auc_test_rounds = 20
    in_feature_map = {
        "Cora":1433,
        "Citeseer":3703,
        "Pubmed":500,
        "BlogCatalog":8189,
        "Flickr":12047,
        "ACM":8337,
        "ogbn-arxiv":128,
    }
    final_args_dict = {
        "dataset": args.dataset,
        "model":{
            "in_feats":in_feature_map[args.dataset],
            "out_feats":args.embedding_dim,
            "global_adg":args.global_adg
        },
        "fit":{
            "device":args.device,
            "batch_size":args.batch_size,
            "num_epoch":args.num_epoch,
            "lr":args.lr,
            "logdir":args.logdir,
            "weight_decay":args.weight_decay,
            "seed":args.seed,
        },
        "predict":{
            "device":args.device,
            "batch_size":args.batch_size,
            "num_workers":args.num_workers,
            "auc_test_rounds":args.auc_test_rounds,
            "logdir":args.logdir
        }
    }
    return final_args_dict


def train_epoch(epoch, loader, net, device, criterion, optimizer):
    """train_epoch [summary]

    [extended_summary]

    Parameters
    ----------
    epoch : [int]
        [epoch number during training]
    loader : [torch.nn.DataLoader]
        [dataloader for training]
    net : [torch.nn.Module]
        [model]
    device : [str]
        [device for training]
    criterion : [type]
        [loss]
    optimizer : [torch.optim.Adam]
        [optimizer for training]
    Returns
    -------
    [type]
        [description]
    """    
    loss_accum = 0
    net.train()
    for step, (pos_subgraph, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph, neg_subgraph = pos_subgraph.to(
            device), neg_subgraph.to(device)
        posfeat = pos_subgraph.ndata['feat'].to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)
        optimizer.zero_grad()
        pos_scores, neg_scores = net(
            pos_subgraph, posfeat, neg_subgraph, negfeat)
        loss = loss_fun(pos_scores, neg_scores, criterion, device)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
    loss_accum /= (step + 1)
    lcprint('TRAIN==>epoch', epoch, 'Average training loss: {:.2f}'.format(
        loss_accum), color='blue')
    return loss_accum


def test_epoch(epoch, loader, net, device, criterion):
    """[summary]
    Parameters:
    -----------
        epoch ([int]]): [epoch number during testing]
        loader ([torch.nn.DataLoader]): [dataloader for testing]
        net ([torch.nn.Module]): [model]
        device ([str]): [device for testing]
        criterion ([torch.nn.Module]): [loss, the same as the loss during training]
    Returns:
    --------
        [numpy.ndarray]: [anomaly score]
    """     
    loss_accum = 0
    net.eval()
    predict_scores = []
    for step, (pos_subgraph, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph, neg_subgraph = pos_subgraph.to(
            device), neg_subgraph.to(device)
        posfeat = pos_subgraph.ndata['feat'].to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)
        pos_scores, neg_scores = net(
            pos_subgraph, posfeat, neg_subgraph, negfeat)
        predict_scores.extend(
            list((torch.sigmoid(neg_scores)-torch.sigmoid(pos_scores)).detach().cpu().numpy()))
        loss = loss_fun(pos_scores, neg_scores, criterion, device)
        loss_accum += loss.item()
    loss_accum /= (step + 1)
    lcprint('VALID==>epoch', epoch, 'Average valid loss: {:.2f}'.format(
        loss_accum), color='blue')
    return np.array(predict_scores)
