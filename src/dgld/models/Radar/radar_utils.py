import shutil
from tqdm import tqdm
from copy import deepcopy
import torch
import dgl
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from utils.common_params import IN_FEATURE_MAP,NUM_NODES_MAP

import argparse
from dgld.utils.common import loadargs_from_json

def set_subargs(parser):
    parser.add_argument('--num_epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.2)
    
def get_subargs(args):
    
    # best_config = loadargs_from_json('src/dgld/config/Radar.json')[args.dataset]
    # config = vars(args)
    # config.update(best_config)
    # args = argparse.Namespace(**config)    
            
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model":{
            
        },
        "fit":{
            "num_epoch": args.num_epoch,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "device": args.device,
        },
        "predict":{
            "device": args.device,
        }
    }
    return final_args_dict, args