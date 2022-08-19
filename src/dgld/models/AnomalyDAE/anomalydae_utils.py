import shutil
import scipy.sparse as sp
import numpy as np
import torch
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def set_subargs(parser):
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='dimension of hidden embedding (default: 256)')
    parser.add_argument('--out_dim', type=int, default=128,
                        help='dimension of output embedding (default: 128)')
    parser.add_argument('--num_epoch', type=int, help='Training epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dropout', type=float,
                        default=0.0, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0, help='weight decay')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='balance parameter')
    parser.add_argument('--eta', type=float, default=5.0,
                        help='Attribute penalty balance parameter')
    parser.add_argument('--theta', type=float, default=40.0,
                        help='structure penalty balance parameter')
    parser.add_argument('--patience', type=int, help='early stop patience',default=10)
    
def get_subargs(args):
    final_args_dict = {
        "dataset": args.dataset,
        "seed":args.seed,
        "model":{
            "feat_size":args.feat_dim,
            "num_nodes":args.num_nodes,
            "embed_dim":args.embed_dim,
            "out_dim":args.out_dim,
            "dropout":args.dropout
        },
        "fit":{
            "lr":args.lr,
            "num_epoch":args.num_epoch,
            "alpha":args.alpha,
            "eta":args.eta,
            "theta":args.theta,
            "device":args.device,
            "patience":args.patience
        },
        "predict":{
            "alpha":args.alpha,
            "eta":args.eta,
            "theta":args.theta,
            "device":args.device,
        }
    }
    return final_args_dict,args


def loss_func(adj, A_hat, attrs, X_hat, alpha,eta, theta):
    # Attribute reconstruction loss
    etas=attrs*(eta-1)+1
    diff_attribute = torch.pow((X_hat - attrs)* etas, 2) 
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    thetas = adj * (theta-1) + 1 
    diff_structure = torch.pow((A_hat - adj)* thetas, 2) 
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + \
        (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost


def train_step(model, optimizer, graph, features,adj_label,alpha,eta,theta):
    model.train()
    optimizer.zero_grad()
    A_hat, X_hat = model(graph, features)
    loss, struct_loss, feat_loss = loss_func(
        adj_label, A_hat, features, X_hat, alpha,eta,theta)
    l = torch.mean(loss)
    l.backward()
    optimizer.step()
    return l, struct_loss, feat_loss,loss.detach().cpu().numpy()


def test_step(model, graph, features,adj_label, alpha,eta,theta):
    model.eval()
    A_hat, X_hat = model(graph, features)
    loss, _, _ = loss_func(adj_label, A_hat, features, X_hat, alpha,eta,theta)
    score = loss.detach().cpu().numpy()
    # print("Epoch:", '%04d' % (epoch), 'Auc', roc_auc_score(label, score))
    return score

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()