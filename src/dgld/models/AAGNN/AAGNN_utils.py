import shutil
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)

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
    final_args_dict = {
        "dataset": args.dataset,
        "seed":args.seed,
        "model":{
            "feat_size":args.feat_dim,
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


