import shutil
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from utils.common_params import IN_FEATURE_MAP


def set_subargs(parser):
    parser.add_argument('--out_feats', type=int, default=256,
                        help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--out_dim', type=int, default=128,
                        help='dimension of output embedding (default: 128)')
    parser.add_argument('--num_epoch', type=int,
                        default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--subgraph_size', type=int, default=4096)
    parser.add_argument('--dropout', type=float,
                        default=0.0, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0, help='weight decay')

    parser.add_argument('--eta', type=float, default=5.0,
                        help='Attribute penalty balance parameter')
    parser.add_argument('--theta', type=float, default=40.0,
                        help='structure penalty balance parameter')

def get_subargs(args):


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

    final_args_dict = {
        "dataset": args.dataset,
        "seed":args.seed,
        "model":{
            "feat_size":IN_FEATURE_MAP[args.dataset],
            "out_feats":args.out_feats
        },
        "fit":{
            "lr":args.lr,
            "num_epoch":args.num_epoch,
            "subgraph_size":args.subgraph_size,
            "device":args.device,
        },
        "predict":{
            "device":args.device,
            "subgraph_size": args.subgraph_size,
        }
    }
    return final_args_dict, args


