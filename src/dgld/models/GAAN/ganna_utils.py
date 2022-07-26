import torch 
import dgl 
import numpy as np
import shutil
import sys
import os
sys.path.append('../../')
import argparse

def set_subargs(parser):
    parser.add_argument('--num_epoch', type=int, help='Training epoch')
    parser.add_argument('--gen_hid_dims', type=list, default=[32,64,128],
                    help='generator hidden dims list')
    parser.add_argument('--ed_hid_dims', type=list, default=[32,64],
                    help='discriminator hidden dims list')
    parser.add_argument('--out_dim', type=int, default=128,
                    help='discriminator of encoder out')
    parser.add_argument('--batch_size', type=int, default=1024,help='batch_size, 0 for all data ')
    parser.add_argument('--g_lr', type=float, default=0.005, help='generator learning rate')
    parser.add_argument('--d_lr', type=float, default=0.005, help='discriminator learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--noise_dim', type=int, default=32, help='noise_dim')
    parser.add_argument('--dropout', type=float,default=0.2, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='balance parameter')
    parser.add_argument('--num_neighbor',type=int,default=200,help='the simple number of neighbor -1 for all neighbor')

def get_subargs(args):
    if args.num_epoch is None:
        if args.dataset in ['Cora', 'Citeseer']:
            args.num_epoch = 100
        elif args.dataset in ['Pubmed','BlogCatalog', 'Flickr', 'ACM']:
            args.num_epoch = 30
        else:
            args.num_epoch = 10
    if args.dataset in ['BlogCatalog','ogbn-arxiv']:
        args.alpha = 0.1
    if args.dataset == 'Flickr':
        args.weight_decay = 0.001
    if args.dataset == 'ACM':
        args.alpha = 0

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
            "noise_dim": args.noise_dim,
            "gen_hid_dims": args.gen_hid_dims,
            "attrb_dim": in_feature_map[args.dataset],
            "ed_hid_dims": args.ed_hid_dims,
            "out_dim": args.out_dim,
            "dropout": args.dropout
        },
        "fit":{
            "num_epoch":args.num_epoch,
            "batch_size":args.batch_size,
            "g_lr":args.g_lr,
            "d_lr":args.d_lr,
            "weight_decay":args.weight_decay,
            "num_neighbor":args.num_neighbor,
            "device":args.device
        },
        "predict":{
            "alpha":args.alpha,
            "batch_size":args.batch_size,
            "num_neighbor":args.num_neighbor,
            "device":args.device
        }
    }
    return final_args_dict,args