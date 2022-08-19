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

    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--num_epoch', type=int, help='Training epoch',default=1000)
    parser.add_argument('--lr', type=float, help='learning rate',default=0.01)
    parser.add_argument('--dropout', type=float,
                        default=0.0, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='balance parameter')
    parser.add_argument('--patience', type=int, help='early stop patience',default=10)


def get_subargs(args):
    final_args_dict = {
        "dataset": args.dataset,
        "seed":args.seed,
        "model":{
            "feat_size":args.feat_dim,
            "hidden_size":args.hidden_dim,
            "dropout":args.dropout
        },
        "fit":{
            "lr":args.lr,
            "num_epoch":args.num_epoch,
            "alpha":args.alpha,
            "device":args.device,
            "patience":args.patience
        },
        "predict":{
            "alpha":args.alpha,
            "device":args.device,
        }
    }
    return final_args_dict,args

def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + \
        (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def train_step( model, optimizer, graph, features, adj_label,alpha):

    model.train()
    optimizer.zero_grad()
    
    A_hat, X_hat = model(graph, features)
    # A_hat, X_hat = model(features,adj)
    loss, struct_loss, feat_loss = loss_func(
        adj_label, A_hat, features, X_hat, alpha)
    l = torch.mean(loss)
    l.backward()
    optimizer.step()
    return l, struct_loss, feat_loss
    # print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "train/struct_loss=", "{:.5f}".format(struct_loss.item()),"train/feat_loss=", "{:.5f}".format(feat_loss.item()))


def test_step(model, graph, features, adj_label,alpha):
    model.eval()
    A_hat, X_hat = model(graph, features)
    # A_hat, X_hat = model(features,adj)
    loss, _, _ = loss_func(adj_label, A_hat, features, X_hat, alpha)
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