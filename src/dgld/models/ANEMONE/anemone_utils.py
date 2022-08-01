import argparse
from tqdm import tqdm
import numpy as np
import torch

import shutil
import sys
import os

sys.path.append('../../')
# from utils.print import cprint, lcprint
from utils.common_params import IN_FEATURE_MAP

def set_subargs(parser):
    """
    get hyperparameter by parser from command line

    Returns
    -------
    final_args_dict : dictionary
        dict of args parser
    """
    parser.add_argument('--lr', type=float)
    parser.add_argument('--alpha', type=float,default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=10)
    parser.add_argument('--auc_test_rounds', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--global_adg', type=bool, default=True)


def get_subargs(args):

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

    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": {
            "in_feats": IN_FEATURE_MAP[args.dataset],
            "out_feats": args.embedding_dim,
            "global_adg": args.global_adg
        },
        "fit": {
            "device": args.device,
            "batch_size": args.batch_size,
            "num_epoch": args.num_epoch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "predict": {
            "device": args.device,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "auc_test_rounds": args.auc_test_rounds,
        }
    }
    return final_args_dict, args



def loss_fun_BCE(pos_scores_rdc, pos_scores_rec, neg_scores_rdc, neg_scores_rec, criterion, device, alpha):
    """
    calculate loss function in Binary CrossEntropy Loss
    Parameters
    ----------
    pos_scores : torch.Tensor
        anomaly score of positive sample
    neg_scores : torch.Tensor
        anomaly score of negative sample
    criterion : torch.nn.Module
        loss function calculation funciton
    device : str
        device for calculation

    Returns
    -------
    loss_accum : torch.Tensor
        loss of single epoch
    """
    scores_rdc = torch.cat([pos_scores_rdc, neg_scores_rdc], dim=0)
    scores_rec = torch.cat([pos_scores_rec, neg_scores_rec], dim=0)
    batch_size = pos_scores_rdc.shape[0]
    pos_label = torch.ones(batch_size).to(device)
    neg_label = torch.zeros(batch_size).to(device)
    labels = torch.cat([pos_label, neg_label], dim=0)
    accuarcy = 0
    loss_rdc = criterion(scores_rdc, labels)
    loss_rec = criterion(scores_rec, labels)
    return (alpha) * torch.mean(loss_rdc) + (1 - alpha) * torch.mean(loss_rec), accuarcy


loss_fun = loss_fun_BCE


def train_epoch(epoch,alpha, loader, net, device, criterion, optimizer):
    """train_epoch, train model in one epoch
    Parameters
    ----------
    epoch : int
        epoch number during training
    loader : torch.nn.DataLoader
        dataloader for training
    net : torch.nn.Module
        model
    device : str
        device for training
    criterion : type
        loss
    optimizer : torch.optim.Adam
        optimizer for training

    Returns
    -------
    loss_accum : torch.Tensor
        loss of single epoch
    """
    loss_accum = 0
    net.train()
    for step, (pos_subgraph, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph, neg_subgraph = pos_subgraph.to(device), neg_subgraph.to(device)
        posfeat = pos_subgraph.ndata['feat'].to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)
        optimizer.zero_grad()
        pos_scores_rdt, pos_scores_rec, neg_scores_rdt, neg_scores_rec = net(pos_subgraph, posfeat, neg_subgraph,
                                                                             negfeat)
        loss, acc = loss_fun(pos_scores_rdt, pos_scores_rec, neg_scores_rdt, neg_scores_rec, criterion, device, alpha)
        # print('loss::::::',loss)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
    loss_accum /= (step + 1)
    return loss_accum


def test_epoch(epoch, alpha, loader, net, device, criterion):
    """test_epoch, test model in one epoch
    Parameters
    ----------
    epoch : int
        epoch number during testin
    loader : torch.nn.DataLoader
        dataloader for testing
    net : torch.nn.Module
        model
    device : str
        device for testing
    criterion : torch.nn.Module
        loss, the same as the loss during training

    Returns
    -------
    predict_scores : numpy.ndarray
        anomaly score
    """
    loss_accum = 0
    net.eval()
    predict_scores = []
    # predict_scores_rec = []
    for step, (pos_subgraph, neg_subgraph) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph, neg_subgraph = pos_subgraph.to(device), neg_subgraph.to(device)
        posfeat = pos_subgraph.ndata['feat'].to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)

        pos_scores_rdt, pos_scores_rec, neg_scores_rdt, neg_scores_rec = net(pos_subgraph, posfeat, neg_subgraph,
                                                                             negfeat)
        predict_scores.extend(list(
            alpha * (torch.sigmoid(neg_scores_rdt) - torch.sigmoid(pos_scores_rdt)).detach().cpu().numpy() + (
                        1 - alpha) * (
                        torch.sigmoid(neg_scores_rec) - torch.sigmoid(pos_scores_rec)).detach().cpu().numpy()))
        loss, acc = loss_fun(pos_scores_rdt, pos_scores_rec, neg_scores_rdt, neg_scores_rec, criterion, device, alpha)
        loss_accum += loss.item()
    loss_accum /= (step + 1)
    # lcprint('VALID==>epoch', epoch, 'Average valid loss: {:.2f}'.format(loss_accum), color='blue')
    # lcprint('Average testing acc: {:.2f}'.format(acc), color='green')
    return np.array(predict_scores)


