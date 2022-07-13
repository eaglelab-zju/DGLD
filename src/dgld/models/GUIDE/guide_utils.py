import torch 
import dgl 
import numpy as np
import shutil
import sys
import os
sys.path.append('../../')

import argparse
def get_parse():
    """
    get hyperparameter by parser from command line

    Returns
    -------
    final_args_dict : dictionary
        dict of args parser
    """
    parser = argparse.ArgumentParser(
        description='Higher-order Structure Based Anomaly Detection on Attributed Networks')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--logdir', type=str, default='tmp')
    parser.add_argument('--attrb_hid', type=int, default=64,
                    help='dimension of hidden embedding for attribute encoder (default: 64)')
    parser.add_argument('--struct_hid', type=int, default=8,
                    help='dimension of hidden embedding for structure encoder (default: 8)')
    parser.add_argument('--num_epoch', type=int, help='Training epoch')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--dropout', type=float,
                        default=0.0, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='balance parameter')
    parser.add_argument('--device', type=str, default='0')

    args = parser.parse_args()

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    if args.lr is None:
        args.lr = 1e-3

    if args.num_epoch is None:
        if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
            args.num_epoch = 200
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
    final_args_dict = {
        "dataset": args.dataset,
        "seed":args.seed,
        "model":{
            "feat_size":in_feature_map[args.dataset],
            "hidden_size":args.hidden_dim,
            "attrb_hidden_size":args.attrb_hid,
            "struct_hidden_size":args.struct_hid,
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

def cal_motifs(nx_g,x,idx):
    """
    count the number of motifs 

    Parameters
    ----------
    nx_g : networkx.Graph
        the graph
    x: int
        the node id you want count
    idx: int
        the motifs id you want count

    Returns
    -------
    the number of motifs
    """
    sum = 0 
    if idx == 0:
        adj_x = list(nx_g[x].keys())
        size = len(adj_x)
        for i in range(size):
            y = adj_x[i]
            adj_y = list(nx_g[y].keys())
            res = set(adj_x) & set(adj_y)
            sum += len(res)
        return sum // 2
    if idx == 1:
        adj_x = list(nx_g[x].keys())
        for i in range(len(adj_x)):
            y = adj_x[i]
            adj_y = set(nx_g[y].keys())
            res = adj_y - set(adj_x)
            sum += len(res) - 1

            res = set(adj_x[i+1:]) - adj_y
            sum += len(res)
        return sum
    if idx == 2:
        size = len(nx_g[x])
        adj_x = list(nx_g[x].keys())
        for i in range(size):
            y = adj_x[i]
            adj_y = set(nx_g[y].keys())
            for j in range(i+1,size):
                z = adj_x[j]
                if z not in adj_y:
                    continue
                adj_z = set(nx_g[z].keys())
                a_list =  adj_y & set(adj_x[j+1:]) & adj_z
                sum += len(a_list)
        return sum 
    if idx == 3:
        size = len(nx_g[x])
        adj_x = list(nx_g[x].keys())
        for i in range(size):
            y = adj_x[i]
            adj_y = set(nx_g[y].keys())

            rx = set(adj_x[i+1:])
            z_set = rx - adj_y
            for z in z_set:
                adj_z = set(nx_g[z].keys())
                res = adj_z & adj_y & set(adj_x)
                sum += len(res)
            z_set = rx & adj_y
            for z in z_set:
                adj_z = set(nx_g[z].keys())
                res = adj_z & adj_y - set(adj_x)
                sum += len(res) - 1

        return sum 
    if idx == 4:
        size = len(nx_g[x])
        adj_x = list(nx_g[x].keys())
        for i in range(size):
            y = adj_x[i]
            adj_y = set(nx_g[y].keys())
            z_list = set(adj_x[i+1:]) - adj_y
            for z in z_list:
                adj_z = set(nx_g[z].keys())
                res = adj_z & adj_y - set(adj_x) 
                sum += len(res) - 1
        return sum 
def get_struct_feat(graph:dgl.DGLGraph):
    """
    Generate the struct feature.Use the number of the motifs to express.
    
    Parameters
    ----------
    graph : DGL.Graph
        input graph

    Returns
    -------
    the struct feature 
    """
    new_g = graph.to_simple()
    new_g = new_g.remove_self_loop()
    nx_g = new_g.to_networkx().to_undirected()
    node_num = new_g.num_nodes()
    struct_feat = np.zeros((node_num,6))
    for i in range(node_num):
        struct_feat[i][5] = len(nx_g[i])
        for  idx in range(5):
            struct_feat[i][idx] = cal_motifs(nx_g,i,idx)
    return torch.tensor(struct_feat,dtype=torch.float32)