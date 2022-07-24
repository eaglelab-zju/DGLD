from ast import arg
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
        description='Generative Adversarial Attributed Network Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--logdir', type=str, default='tmp')
    parser.add_argument('--gen_hid_dims', type=list, default=[32,64,128],
                    help='generator hidden dims list')
    parser.add_argument('--ed_hid_dims', type=list, default=[32,64],
                    help='discriminator hidden dims list')
    parser.add_argument('--out_dim', type=int, default=128,
                    help='discriminator of encoder out')
    parser.add_argument('--batch_size', type=int, default=0,help='batch_size, 0 for all data ')
    parser.add_argument('--num_epoch', type=int, help='Training epoch')
    parser.add_argument('--g_lr', type=float, default=0.005, help='generator learning rate')
    parser.add_argument('--d_lr', type=float, default=0.005, help='discriminator learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--noise_dim', type=int, default=32, help='noise_dim')
    parser.add_argument('--dropout', type=float,
                        default=0., help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='balance parameter')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--num_neighbor',type=int,default=-1,help='the simple number of neighbor -1 for all neighbor')
    args = parser.parse_args()

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)


    if args.num_epoch is None:
        if args.dataset in ['Cora', 'Citeseer', 'Pubmed','BlogCatalog', 'Flickr', 'ACM']:
            args.num_epoch = 100
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
            "noise_dim":args.noise_dim,
            "gen_hid_dims":args.gen_hid_dims,
            "attrb_dim":in_feature_map[args.dataset],
            "ed_hid_dims":args.ed_hid_dims,
            "out_dim":args.out_dim,
            "dropout":args.dropout
        },
        "fit":{
            "batch_size":args.batch_size,
            "g_lr":args.g_lr,
            "d_lr":args.d_lr,
            "logdir":args.logdir,
            "num_epoch":args.num_epoch,
            "weight_decay":args.weight_decay,
            "num_neighbor":args.num_neighbor,
            "alpha":args.alpha,
            "device":args.device,
        },
        "predict":{
            "batch_size":args.batch_size,
            "alpha":args.alpha,
            "device":args.device,
        }
    }
    return final_args_dict
