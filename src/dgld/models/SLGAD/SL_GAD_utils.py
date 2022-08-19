from email.policy import default
from tqdm import tqdm
import numpy as np 
import torch
import shutil
from datetime import datetime
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from utils.common import lcprint


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

def set_subargs(parser):

    parser.add_argument('--num_epoch', type=int, help='Training epoch')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int, default=4)
    parser.add_argument('--auc_test_rounds', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--global_adg', type=lambda x: x.lower() == 'true', default=True)  
    parser.add_argument('--alpha', type = float, default = 1.0)
    parser.add_argument('--beta', type = float, default = 0.6)
    parser.add_argument('--act_function', type = str, default= "PReLU")
    parser.add_argument('--degree_coefficient', type=float, default=-1)
    parser.add_argument('--attention', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--positive_subgraph_cor', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--negative_subgraph_cor', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--arw', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--patience', type=int, default=400)
    parser.add_argument('--expid', type=int)

def get_subargs(args):
    final_args_dict = {
        "dataset": args.dataset,
        "seed":args.seed,
        "model":{
            "in_feats":args.feat_dim,
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
            "weight_decay":args.weight_decay,
        },
        "predict":{
            "device":args.device,
            "batch_size":args.batch_size,
            "num_workers":args.num_workers,
            "auc_test_rounds":args.auc_test_rounds,
        }
    }
    return final_args_dict, args


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

    start_time = datetime.now()
    end_time = datetime.now()
    for step, (pos_subgraph_1, pos_subgraph_2, neg_subgraph, idx) in enumerate(tqdm(loader, desc="Iteration")):
        end_time = datetime.now()
        pos_subgraph_1 = pos_subgraph_1.to(device)
        pos_subgraph_2 = pos_subgraph_2.to(device)
        pos_subgraph = [pos_subgraph_1, pos_subgraph_2] # .to(device)
        neg_subgraph = neg_subgraph.to(device)
        posfeat_1 = pos_subgraph_1.ndata['feat'].to(device)
        posfeat_2 = pos_subgraph_2.ndata['feat'].to(device)
        raw_posfeat_1 = pos_subgraph_1.ndata['Raw_Feat']
        raw_posfeat_2 = pos_subgraph_2.ndata['Raw_Feat']

        posfeat = [posfeat_1, posfeat_2, raw_posfeat_1, raw_posfeat_2] # .to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)
        optimizer.zero_grad()
        
        loss, single_predict_scores, single_contrastive_loss, single_generative_loss = net(pos_subgraph, posfeat, neg_subgraph, negfeat, args)
        loss.backward()
        optimizer.step()
        loss_accum += loss.cpu().detach().numpy() * pos_subgraph_1.number_of_nodes()
        contrastive_loss_accum += single_contrastive_loss.cpu().detach().numpy() * pos_subgraph_1.number_of_nodes()
        generative_loss_accum += single_generative_loss.cpu().detach().numpy() * pos_subgraph_1.number_of_nodes()

        number_nodes = number_nodes + pos_subgraph_1.number_of_nodes()
        start_time = datetime.now()

    loss_accum /= number_nodes
    contrastive_loss_accum /= number_nodes
    generative_loss_accum /= number_nodes
    lcprint('TRAIN==>epoch', epoch, 'Average training loss: {:.8f}'.format(loss_accum), color='blue')
    print('contrastive loss : ', contrastive_loss_accum)
    print('generative loss : ', generative_loss_accum)
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
    number_nodes = 0
    predict_scores = torch.full([loader.dataset.dataset.number_of_nodes(), ], 0.0)

    for step, (pos_subgraph_1, pos_subgraph_2, neg_subgraph, idx) in enumerate(tqdm(loader, desc="Iteration")):
        pos_subgraph_1 = pos_subgraph_1.to(device)
        pos_subgraph_2 = pos_subgraph_2.to(device)
        pos_subgraph = [pos_subgraph_1, pos_subgraph_2] # .to(device)
        neg_subgraph = neg_subgraph.to(device)
        posfeat_1 = pos_subgraph_1.ndata['feat'].to(device)
        posfeat_2 = pos_subgraph_2.ndata['feat'].to(device)
        raw_posfeat_1 = pos_subgraph_1.ndata['Raw_Feat']
        raw_posfeat_2 = pos_subgraph_2.ndata['Raw_Feat']
        posfeat = [posfeat_1, posfeat_2, raw_posfeat_1, raw_posfeat_2] # .to(device)
        negfeat = neg_subgraph.ndata['feat'].to(device)

        loss, single_predict_scores, single_contrastive_loss, single_generative_loss = net(pos_subgraph, posfeat, neg_subgraph, negfeat, args)
        predict_scores[idx] = single_predict_scores.squeeze().to('cpu')
        loss_accum += loss.item() * pos_subgraph_1.number_of_nodes()
        contrastive_loss_accum += single_contrastive_loss.item() * pos_subgraph_1.number_of_nodes()
        generative_loss_accum += single_generative_loss.item() * pos_subgraph_1.number_of_nodes()
        
        number_nodes = number_nodes + pos_subgraph_1.number_of_nodes()

    predict_scores = predict_scores.tolist()
    loss_accum /= number_nodes
    contrastive_loss_accum /= number_nodes
    generative_loss_accum /= number_nodes
    
    lcprint('VALID==>epoch', epoch, 'Average valid loss: {:.8f}'.format(loss_accum), color='blue')
    print('contrastive loss : ', contrastive_loss_accum)
    print('generative loss : ', generative_loss_accum)
    return np.array(torch.tensor(predict_scores, device = 'cpu'))

    
