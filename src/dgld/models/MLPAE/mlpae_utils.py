import shutil
from tqdm import tqdm
from copy import deepcopy
import torch
import dgl
import os, sys

current_file_name = __file__
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from utils.common_params import IN_FEATURE_MAP


def set_subargs(parser):
    parser.add_argument('--num_epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)


def get_subargs(args):
    if args.dataset == 'Citeseer':
        args.n_layers = 3
    if args.dataset == 'Pubmed':
        args.hidden_dim = 256

    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": {
            "feat_size": IN_FEATURE_MAP[args.dataset],
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "dropout": args.dropout
        },
        "fit": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_epoch": args.num_epoch,
            "weight_decay": args.weight_decay,
            "device": args.device
        },
        "predict": {
            "batch_size": args.batch_size,
            "device": args.device
        }
    }
    return final_args_dict, args
