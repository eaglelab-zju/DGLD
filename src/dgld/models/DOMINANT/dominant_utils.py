import shutil
import sys
import scipy.sparse as sp
import os
sys.path.append('../../')

import argparse
from tqdm import tqdm
import numpy as np
import torch


def get_parse():
    parser = argparse.ArgumentParser(
        description='Deep Anomaly Detection on Attributed Networks')
    # "Cora", "Pubmed", "Citeseer"
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=2022)
    # max min avg  weighted_sum
    parser.add_argument('--logdir', type=str, default='tmp')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--num_epoch', type=int, help='Training epoch')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--dropout', type=float,
                        default=0.0, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='balance parameter')
    parser.add_argument('--device', type=str, default='0')

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

    if args.dataset == 'Citeseer':
        args.alpha = 0.8
        args.seed = 4096
        args.dropout = 0.3
        args.hidden_dim = 32
    elif args.dataset == 'Pubmed':
        args.alpha = 0.8
        args.seed = 4096
        args.dropout = 0.3
        args.hidden_dim = 128
    elif args.dataset == 'Flickr':
        args.alpha = 0.6
        args.seed = 1024
        args.dropout = 0.0
        args.hidden_dim = 64
    elif args.dataset == 'ACM':
        args.alpha = 0.2
        args.seed = 4096
        args.dropout = 0.0
        args.hidden_dim = 16
        # args.lr = 1e-5
        args.num_epoch = 300


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
        "seed":args.seed,
        "model":{
            "feat_size":in_feature_map[args.dataset],
            "hidden_size":args.hidden_dim,
            "dropout":args.dropout
        },
        "fit":{
            "lr":args.lr,
            "logdir":args.logdir,
            "num_epoch":args.num_epoch,
            "alpha":args.alpha,
            "device":args.device,
        },
        "predict":{
            "alpha":args.alpha,
            "device":args.device,
        }
    }
    return final_args_dict


def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + \
        (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost


def train_step( model, optimizer, graph, features, adj_label,alpha):

    model.train()
    optimizer.zero_grad()
    
    A_hat, X_hat = model(graph, features)
    # A_hat, X_hat = model(features,adj)
    loss, struct_loss, feat_loss = loss_func(
        adj_label, A_hat, features, X_hat, alpha)
    l = torch.mean(loss)
    l.backward()
    optimizer.step()
    return l, struct_loss, feat_loss
    # print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))


def test_step(model, graph, features, adj_label,alpha):
    model.eval()
    A_hat, X_hat = model(graph, features)
    # A_hat, X_hat = model(features,adj)
    loss, _, _ = loss_func(adj_label, A_hat, features, X_hat, alpha)
    score = loss.detach().cpu().numpy()
    # print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))
    return score

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()