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


def set_subargs(parser):
    parser.add_argument('--num_epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden embedding')
    parser.add_argument('--n_layers', type=int, default=3, help='num of mlp layers')


def get_subargs(args):
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": {
            "feat_size": args.feat_dim,
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
