import dgl 
import torch 
import torch.nn.functional as F 
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from utils.common_params import NUM_NODES_MAP,IN_FEATURE_MAP

from tqdm import tqdm 

def set_subargs(parser):
    parser.add_argument('--num_epoch', type=int, help='Training epoch')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='balance parameter')
    parser.add_argument('--gamma', type=float, default=0.09,
                        help='balance parameter')
    parser.add_argument('--lr', type=float,default = 0.01, help='learning rate')
def get_subargs(args):
    if args.num_epoch is None:
        args.num_epoch = 2000
    if args.dataset in ['Cora','Citeseer','BlogCatalog']:
        args.gamma = 0.001
    if args.dataset == 'Pubmed':
        args.gamma = 0.1
    if args.dataset == 'Flickr':
        args.lr = 0.05
        args.gamma = 0.0001

    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model":{
            
        },
        "fit":{
            "num_epoch":args.num_epoch,
            "lr":args.lr,
            "device":args.device,
            "gamma":args.gamma,
            "weight_decay":args.weight_decay,
        },
        "predict":{
        }
    }
    return final_args_dict,args


def Laplacian(G : dgl.DGLGraph):
    D = G.in_degrees().float()
    D = torch.diag(D)
    return D - G.adj()

