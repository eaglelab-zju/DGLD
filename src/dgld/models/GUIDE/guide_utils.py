import torch 
import dgl 
import numpy as np
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from tqdm import tqdm

def set_subargs(parser):
    parser.add_argument('--attrb_hid', type=int, default=64,
                    help='dimension of hidden embedding for attribute encoder (default: 64)')
    parser.add_argument('--struct_hid', type=int, default=32,
                    help='dimension of hidden embedding for structure encoder (default: 8)')
    parser.add_argument('--num_epoch', type=int, help='Training epoch')
    parser.add_argument('--lr', type=float,default = 0.01, help='learning rate')
    parser.add_argument('--dropout', type=float,
                        default=0.5, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='balance parameter')
    parser.add_argument('--struct_dim', type=int, default=6,
                        help='struct feature dim')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='the number of layers')
    parser.add_argument('--batch_size', type=int, default=0,help='batch_size, 0 for all data ')

def get_subargs(args):  
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model":{
            "attrb_dim": args.feat_dim,
            "struct_dim": args.struct_dim,
            "num_layers": args.num_layers,
            "attrb_hid": args.attrb_hid,
            "struct_hid": args.struct_hid,
            "dropout": args.dropout,
        },
        "fit":{
            "num_epoch":args.num_epoch,
            "batch_size":args.batch_size,
            "lr":args.lr,
            "alpha":args.alpha,
            "device":args.device
        },
        "predict":{
            "alpha":args.alpha,
            "batch_size":args.batch_size,
            "device":args.device
        }
    }
    return final_args_dict,args

def cal_motifs(nx_g,x,idx,sample=50):
    """
    count the number of motifs 

    Parameters
    ----------
    nx_g : networkx.Graph
        the graph
    x: int
        the node id you want count
    idx: int
        the motifs id you want count

    Returns
    -------
    the number of motifs
    """
    sum = 0 
    if idx == 0:
        adj_x = list(nx_g[x].keys())
        if(len(adj_x) > sample):
            adj_x = np.random.choice(adj_x, size=sample, replace=False)
        size = len(adj_x)
        for i in range(size):
            y = adj_x[i]
            adj_y = list(nx_g[y].keys())
            if(len(adj_y) > sample):
                adj_y = np.random.choice(adj_y, size=sample, replace=False)
            res = set(adj_x) & set(adj_y)
            sum += len(res)
        return sum // 2
    if idx == 1:
        adj_x = list(nx_g[x].keys())
        if(len(adj_x) > sample):
            adj_x = np.random.choice(adj_x, size=sample, replace=False)
        for i in range(len(adj_x)):
            y = adj_x[i]
            adj_y = list(nx_g[y].keys())
            if(len(adj_y) > sample):
                adj_y = np.random.choice(adj_y, size=sample, replace=False)
            adj_y = set(adj_y)
            res = adj_y - set(adj_x)
            sum += len(res) - 1

            res = set(adj_x[i+1:]) - adj_y
            sum += len(res)
        return sum
    if idx == 2:
        adj_x = list(nx_g[x].keys())
        if(len(adj_x) > sample):
            adj_x = np.random.choice(adj_x, size=sample, replace=False)
        size = len(adj_x)
        for i in range(size):
            y = adj_x[i]
            adj_y = list(nx_g[y].keys())
            if(len(adj_y) > sample):
                adj_y = np.random.choice(adj_y, size=sample, replace=False)
            adj_y = set(adj_y)
            for j in range(i+1,size):
                z = adj_x[j]
                if z not in adj_y:
                    continue
                adj_z = set(nx_g[z].keys())
                if(len(adj_z) > sample):
                    adj_z = set(np.random.choice(list(adj_z), size=sample, replace=False))
                a_list =  adj_y & set(adj_x[j+1:]) & adj_z
                sum += len(a_list)
        return sum 
    if idx == 3:
        adj_x = list(nx_g[x].keys())
        if(len(adj_x) > sample):
            adj_x = np.random.choice(adj_x, size=sample, replace=False)
        size = len(adj_x)
        for i in range(size):
            y = adj_x[i]
            adj_y = set(nx_g[y].keys())
            if(len(adj_y) > sample):
                adj_y = set(np.random.choice(list(adj_y), size=sample, replace=False))
            rx = set(adj_x[i+1:])
            z_set = rx - adj_y
            for z in z_set:
                adj_z = set(nx_g[z].keys())
                if(len(adj_z) > sample):
                    adj_z = set(np.random.choice(list(adj_z), size=sample, replace=False))
                res = adj_z & adj_y & set(adj_x)
                sum += len(res)
            z_set = rx & adj_y
            for z in z_set:
                adj_z = set(nx_g[z].keys())
                if(len(adj_z) > sample):
                    adj_z = set(np.random.choice(list(adj_z), size=sample, replace=False))
                res = adj_z & adj_y - set(adj_x)
                sum += len(res) - 1

        return sum 
    if idx == 4:
        adj_x = list(nx_g[x].keys())
        if(len(adj_x) > sample):
            adj_x = np.random.choice(adj_x, size=sample, replace=False)
        size = len(adj_x)
        for i in range(size):
            y = adj_x[i]
            adj_y = set(nx_g[y].keys())
            if(len(adj_y) > sample):
                adj_y = set(np.random.choice(list(adj_y), size=sample, replace=False))
            z_list = set(adj_x[i+1:]) - adj_y
            for z in z_list:
                adj_z = set(nx_g[z].keys())
                if(len(adj_z) > sample):
                    adj_z = set(np.random.choice(list(adj_z), size=sample, replace=False))
                res = adj_z & adj_y - set(adj_x) 
                sum += len(res) - 1
        return sum 
def get_struct_feat(graph:dgl.DGLGraph):
    """
    Generate the struct feature.Use the number of the motifs to express.
    
    Parameters
    ----------
    graph : DGL.Graph
        input graph

    Returns
    -------
    the struct feature 
    """
    print("generate struct feature")
    new_g = graph.to_simple()
    new_g = new_g.remove_self_loop()
    nx_g = new_g.to_networkx().to_undirected()
    node_num = new_g.num_nodes()
    struct_feat = np.zeros((node_num,6))
    for i in tqdm(range(node_num)):
        struct_feat[i][5] = len(nx_g[i])
        for  idx in range(5):
            struct_feat[i][idx] = cal_motifs(nx_g,i,idx)
    return torch.tensor(struct_feat,dtype=torch.float32)