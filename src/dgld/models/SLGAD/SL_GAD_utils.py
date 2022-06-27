import argparse
from tqdm import tqdm
import numpy as np 
import torch

import shutil
import sys
import os
sys.path.append('../../')
from utils.print import cprint, lcprint
from datetime import datetime

# torch.set_printoptions(precision=8)
# torch.set_default_tensor_type(torch.DoubleTensor)

def loss_fun_BPR(pos_scores, neg_scores, criterion, device):
    """
    The function to compute loss function of Bayesian personalized ranking

    Parameters
    ----------
    pos_scores : Tensor.tensor
        anomaly score of postive subgraph
    neg_scores : Torch.tensor
        anomaly score of negative subgrph
    criterion : torch.nn.Functions
        functions to compute loss
    device : string
        device of computation
    
    Returns
    -------
    L : Torch.tensor
        loss
    """
    batch_size = pos_scores.shape[0]
    labels = torch.ones(batch_size).to(device)
    return criterion(pos_scores-neg_scores, labels)

def loss_fun_BCE(pos_scores, neg_scores, criterion, device):
    """
    The function to compute loss function of Bayesian personalized ranking

    Parameters
    ----------
    pos_scores : Tensor.tensor
        anomaly score of postive subgraph
    neg_scores : Torch.tensor
        anomaly score of negative subgrph
    criterion : torch.nn.Functions
        functions to compute loss
    device : string
        device of computation
    
    Returns
    -------
    L : Torch.tensor
        loss
    """
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    batch_size = pos_scores.shape[0]
    pos_label = torch.ones(batch_size).to(device)
    neg_label = torch.zeros(batch_size).to(device)
    labels = torch.cat([pos_label, neg_label], dim=0)
    return criterion(scores, labels)

loss_fun = loss_fun_BCE
def get_parse():
    """
    The function to get dictionary of parser

    Parameters
    ----------
    None

    Returns
    -------
    None

    final_args_dict : dictionary
        the dictionary of arg parser
    """
    # parser = argparse.ArgumentParser(description='CoLA: Self-Supervised Contrastive Learning for Anomaly Detection')
    parser = argparse.ArgumentParser(description = 'Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='Cora')  # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=4)
    # parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='tmp')  
    parser.add_argument('--global_adg', type=bool, default=True)  
    parser.add_argument('--alpha', type = float, default = 1.0)
    parser.add_argument('--beta', type = float, default = 0.6)
    parser.add_argument('--act_function', type = str, default= "PReLU")
    parser.add_argument('--degree_coefficient', type=float, default=-1)
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--positive_subgraph_cor', type=bool, default=False)
    parser.add_argument('--negative_subgraph_cor', type=bool, default=False)
    parser.add_argument('--arw', type=bool, default=False)
    parser.add_argument('--patience', type=int, default=400)
    parser.add_argument('--expid', type=int)

    args = parser.parse_args()
    assert args.expid is not None, "experiment id needs to be assigned."

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)
    else:
        os.makedirs(args.logdir)

    if args.lr is None:
        if args.dataset in ['Cora','Citeseer','Pubmed','Flickr']:
            args.lr = 1e-3
        elif args.dataset == 'ACM':
            args.lr = 5e-4
        elif args.dataset == 'BlogCatalog':
            args.lr = 3e-3
        elif args.dataset == 'ogbn-arxiv':
            args.lr = 1e-3

    if args.num_epoch is None:
        if args.dataset in ['Cora','Citeseer','Pubmed']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog','Flickr','ACM']:
            args.num_epoch = 400
        else:
            args.num_epoch = 20
    
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
        "Flickr":12407,
        "ACM":8337,
        "ogbn-arxiv":128,
    }
    
    final_args_dict = {
        "dataset": args.dataset,
        "model":{
            "in_feats":in_feature_map[args.dataset],
            "out_feats":args.embedding_dim,
            "global_adg":args.global_adg,
            "alpha":args.alpha,
            "beta":args.beta,
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
    return args

# import Time_Process
import time
def train_epoch(epoch, args, loader, net, device, criterion, optimizer):
    """
    The function to train

    Parameters
    ----------
    epoch : int
        the number of epoch to train
    loader : GraphDataLoader
        get subgraph set
    net : class
        model
    device : string
        device of computation
    criterion : torch.nn.Functions
        functions to compute loss
    optimizer : optim.Adam
        optimizer to adjust model
    
    Returns
    -------
    L : Torch.tensor
        loss
    """
    loss_accum = 0
    contrastive_loss_accum = 0
    generative_loss_accum = 0
    net.train()
    number_nodes = 0
    # last_time = time.time()
    # Time_Process.global_time.update_Time()
    # print("done", time.time())
    # print(Time_Process.global_time.last_time)
    start_time = datetime.now()
    end_time = datetime.now()
    for step, (pos_subgraph_1, pos_subgraph_2, neg_subgraph, idx) in enumerate(tqdm(loader, desc="Iteration")):
        end_time = datetime.now()
        # print('sample time : ', end_time - start_time)
        # exit()
        # print(pos_subgraph_1)
        # Time_Process.global_time.process_Time("before preprocess")
        # new_time = time.time()
        # print("before preprocess", time.time())
        # print("enumerate", new_time - last_time)
        # last_time = new_time
        # print("pos_subgraph_1", pos_subgraph_1)
        # print(pos_subgraph_1)
        # print(pos_subgraph_1.nodes())
        # exit()

        # print(pos_subgraph_1[:20])
        # print(pos_subgraph_2[:20])
        # print(idx)
        # exit()

        pos_subgraph_1 = pos_subgraph_1.to(device)
        pos_subgraph_2 = pos_subgraph_2.to(device)
        pos_subgraph = [pos_subgraph_1, pos_subgraph_2] # .to(device)
        neg_subgraph = neg_subgraph.to(device)
        posfeat_1 = pos_subgraph_1.ndata['feat'].to(device)
        posfeat_2 = pos_subgraph_2.ndata['feat'].to(device)
        raw_posfeat_1 = pos_subgraph_1.ndata['Raw_Feat']
        raw_posfeat_2 = pos_subgraph_2.ndata['Raw_Feat']
        # print(torch.sum(raw_posfeat_1, dim = 1))
        # print(posfeat_1.dtype)
        # exit()
        posfeat = [posfeat_1, posfeat_2, raw_posfeat_1, raw_posfeat_2] # .to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)
        optimizer.zero_grad()
        
        # pos_scores_1, pos_scores_2, neg_scores = net(pos_subgraph_1, pos_subgraph_2, posfeat_1, posfeat_2, neg_subgraph, negfeat)
        # pos_scores, neg_scores = net(pos_subgraph, posfeat, neg_subgraph, negfeat)
        
        # loss = loss_fun(pos_scores_1, pos_scores_2, neg_scores, criterion, device)
        # loss = loss_fun(pos_scores, neg_scores, criterion, device)
        
        # new_time = time.time()
        # print("before net", time.time())
        # print("before net", new_time - last_time)
        # last_time = new_time
        # Time_Process.global_time.process_Time("before net")

        loss, single_predict_scores, single_contrastive_loss, single_generative_loss = net(pos_subgraph, posfeat, neg_subgraph, negfeat, args)
        # new_time = time.time()
        # print("before backward", time.time())
        # print("before backward", new_time - last_time)
        # last_time = new_time
        # Time_Process.global_time.process_Time("before backward")
        loss.backward()
        optimizer.step()
        # loss_accum += loss.item()
        # print("done", time.time())
        # Time_Process.global_time.process_Time("done")
        # print(loss.dtype)
        # print(type(loss.item()))
        # print(loss.item().dtype)
        # exit()
        loss_accum += loss.cpu().detach().numpy() * pos_subgraph_1.number_of_nodes()
        contrastive_loss_accum += single_contrastive_loss.cpu().detach().numpy() * pos_subgraph_1.number_of_nodes()
        generative_loss_accum += single_generative_loss.cpu().detach().numpy() * pos_subgraph_1.number_of_nodes()

        number_nodes = number_nodes + pos_subgraph_1.number_of_nodes()
        start_time = datetime.now()

    loss_accum /= number_nodes
    contrastive_loss_accum /= number_nodes
    generative_loss_accum /= number_nodes
    # print(type(loss_accum))
    # print(loss_accum.dtype)
    # exit()
    lcprint('TRAIN==>epoch', epoch, 'Average training loss: {:.8f}'.format(loss_accum), color='blue')
    print('contrastive loss : ', contrastive_loss_accum)
    print('generative loss : ', generative_loss_accum)
    # exit()
    return loss_accum

def test_epoch(epoch, args, loader, net, device, criterion, optimizer):
    """
    The function to train

    Parameters
    ----------
    epoch : int
        the number of epoch to train
    loader : GraphDataLoader
        get subgraph set
    net : class
        model
    device : string
        device of computation
    criterion : torch.nn.Functions
        functions to compute loss
    optimizer : optim.Adam
        optimizer to adjust model

    Returns
    -------
    predict_scores : Torch.tensor
        anomaly scores of anchor nodes
    """
    loss_accum = 0
    contrastive_loss_accum = 0
    generative_loss_accum = 0
    net.eval()
    # predict_scores = []
    number_nodes = 0
    predict_scores = torch.full([loader.dataset.dataset.number_of_nodes(), ], 0.0)

    for step, (pos_subgraph_1, pos_subgraph_2, neg_subgraph, idx) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph_1 = pos_subgraph_1.to(device)
        pos_subgraph_2 = pos_subgraph_2.to(device)
        pos_subgraph = [pos_subgraph_1, pos_subgraph_2] # .to(device)
        neg_subgraph = neg_subgraph.to(device)
        posfeat_1 = pos_subgraph_1.ndata['feat'].to(device)
        # print(posfeat_1[:20, :5])
        # exit()
        posfeat_2 = pos_subgraph_2.ndata['feat'].to(device)
        raw_posfeat_1 = pos_subgraph_1.ndata['Raw_Feat']
        raw_posfeat_2 = pos_subgraph_2.ndata['Raw_Feat']
        posfeat = [posfeat_1, posfeat_2, raw_posfeat_1, raw_posfeat_2] # .to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)

        loss, single_predict_scores, single_contrastive_loss, single_generative_loss = net(pos_subgraph, posfeat, neg_subgraph, negfeat, args)
        # predict_scores.extend([idx.tolist(), single_predict_scores.tolist()])
        predict_scores[idx] = single_predict_scores.squeeze().to('cpu')
        loss_accum += loss.item() * pos_subgraph_1.number_of_nodes()
        contrastive_loss_accum += single_contrastive_loss.item() * pos_subgraph_1.number_of_nodes()
        generative_loss_accum += single_generative_loss.item() * pos_subgraph_1.number_of_nodes()
        
        number_nodes = number_nodes + pos_subgraph_1.number_of_nodes()

    predict_scores = predict_scores.tolist()
    # print(predict_scores)
    # exit()
    loss_accum /= number_nodes
    contrastive_loss_accum /= number_nodes
    generative_loss_accum /= number_nodes
    
    lcprint('VALID==>epoch', epoch, 'Average valid loss: {:.8f}'.format(loss_accum), color='blue')
    print('contrastive loss : ', contrastive_loss_accum)
    print('generative loss : ', generative_loss_accum)
    # print(predict_scores[:10])
    # exit()
    # print(len(predict_scores))
    # exit()
    # print(torch.tensor(predict_scores)[:20])
    # exit()
    return np.array(torch.tensor(predict_scores, device = 'cpu'))

    
