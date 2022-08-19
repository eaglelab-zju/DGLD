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
    parser.add_argument('--eps', type=float, default=0.5, help='neighborhood threshold')
    parser.add_argument('--mu', type=int, default=2, help='minimal size of clusters')


def get_subargs(args):
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": {
            "eps": args.eps,
            "mu": args.mu
        },
        "fit": {
        },
        "predict": {
        }
    }
    return final_args_dict, args
