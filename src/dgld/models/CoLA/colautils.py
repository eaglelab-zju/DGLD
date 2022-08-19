from tqdm import tqdm
import numpy as np
import torch
import shutil
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from utils.common import lcprint

def loss_fun_BPR(pos_scores, neg_scores, criterion, device):
    """
    calculate loss function in Bayesian Personalized Ranking

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
    batch_size = pos_scores.shape[0]
    labels = torch.ones(batch_size).to(device)
    return criterion(pos_scores-neg_scores, labels)


def loss_fun_BCE(pos_scores, neg_scores, criterion, device):
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
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    batch_size = pos_scores.shape[0]
    pos_label = torch.ones(batch_size).to(device)
    neg_label = torch.zeros(batch_size).to(device)
    labels = torch.cat([pos_label, neg_label], dim=0)
    return criterion(scores, labels)


loss_fun = loss_fun_BCE


def set_subargs(parser):
    """
    get hyperparameter by parser from command line

    Returns
    -------
    final_args_dict : dictionary
        dict of args parser
    """
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=4)
    parser.add_argument('--auc_test_rounds', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--global_adg', type=lambda x: x.lower() == 'true', default=True)

def get_subargs(args):    
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model":{
            "in_feats":args.feat_dim,
            "out_feats":args.embedding_dim,
            "global_adg":args.global_adg
        },
        "fit":{
            "device":args.device,
            "batch_size":args.batch_size,
            "num_epoch":args.num_epoch,
            "lr":args.lr,
            "weight_decay":args.weight_decay,
            "seed":args.seed,
        },
        "predict":{
            "device":args.device,
            "batch_size":args.batch_size,
            "num_workers":args.num_workers,
            "auc_test_rounds":args.auc_test_rounds,
        }
    }
    return final_args_dict,args


def train_epoch(epoch, loader, net, device, criterion, optimizer):
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
