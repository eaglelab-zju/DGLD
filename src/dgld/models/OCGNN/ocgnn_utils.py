import argparse
import shutil
import sys
import os
sys.path.append('../../')


def get_parse():
    """
    get hyperparameter by parser from command line

    Returns
    -------
    final_args_dict : dictionary
        dict of args parser
    """
    parser = argparse.ArgumentParser(
        description="One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks")
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="Cora/Citeseer/Pubmed/BlogCatalog/Flickr/ACM/ogbn-arxiv")
    parser.add_argument("--logdir", type=str, default="tmp")
    parser.add_argument("--seed", type=int, default=52,
                        help="random seed, -1 means dont fix seed")
    parser.add_argument("--device", type=str, default="0",
                        help="device")
    parser.add_argument("--module", type=str, default="GCN",
                        help="GCN/GAT/GraphSAGE")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="dropout probability")
    parser.add_argument("--batchsize", type=int, default=0,
                        help='batch size')
    parser.add_argument("--nu", type=float, default=0.4,
                        help="hyperparameter nu (must be 0 < nu <= 1)")
    parser.add_argument("--n-worker", type=int, default=1,
                        help="number of workers when loading data")
    parser.add_argument("--normal-class", type=int, default=0,
                        help="normal class")
    parser.add_argument("--num_epoch", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="dimension of hidden embedding (default: 32)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="number of hidden gnn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-3,
                        help="Weight for L2 loss")
    parser.add_argument("--warmup_epoch", type=int, default=1,
                        help="number of warmup epoch")

    args = parser.parse_args()

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    in_feature_map = {
        "Cora": 1433,
        "Citeseer": 3703,
        "Pubmed": 500,
        "BlogCatalog": 8189,
        "Flickr": 12047,
        "ACM": 8337,
        "ogbn-arxiv": 128,
    }
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": {
            "feat_size": in_feature_map[args.dataset],
            "module": args.module,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "nu": args.nu
        },
        "fit": {
            "lr": args.lr,
            "batch_size": args.batchsize,
            "num_epoch": args.num_epoch,
            "warmup_epoch": args.warmup_epoch,
            "log_dir": args.logdir,
            "weight_decay": args.weight_decay,
            "device": args.device
        },
        "predict": {
            "batch_size": args.batchsize,
            "device": args.device
        }
    }
    return final_args_dict
