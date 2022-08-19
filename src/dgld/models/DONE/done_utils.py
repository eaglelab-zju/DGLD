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

def set_subargs(parser):
    parser.add_argument('--num_epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help="weight decay (L2 penalty)")
    parser.add_argument('--dropout', type=float, default=0., help="rate of dropout")
    parser.add_argument('--batch_size', type=int, default=0, help="size of training batch")
    parser.add_argument('--max_len', type=int, default=0, help="maximum length of the truncated random walk")
    parser.add_argument('--restart', type=float, default=0., help="probability of restart")
    parser.add_argument('--num_neighbors', type=int, default=-1, help="number of sampling neighbors")
    parser.add_argument('--embedding_dim', type=int, default=32, help="dimension of embedding")
    
def get_subargs(args):
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model":{
            "feat_size": args.feat_dim,
            "num_nodes": args.num_nodes,
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
        },
        "fit":{
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_epoch": args.num_epoch,
            "device": args.device,
            "batch_size": args.batch_size,
            "num_neighbors": args.num_neighbors,
            "max_len": args.max_len, 
            "restart": args.restart,
        },
        "predict":{
            "device": args.device,
            "batch_size": args.batch_size,
            "max_len": args.max_len, 
            "restart": args.restart,
        }
    }
    return final_args_dict, args


def random_walk_with_restart(g:dgl.DGLGraph, k=3, r=0.3, eps=1e-5):
    """Consistent with the description of "Network Preprocessing" in Section 4.1 of the paper.

    Parameters
    ----------
    g : dgl.DGLGraph
        graph data
    k : int, optional
        The maximum length of the truncated random walk, by default 3
    r : float, optional
        Probability of restart, by default 0.3
    eps : float, optional
        To avoid errors when the reciprocal of a node's out-degree is inf, by default 0.1

    Returns
    -------
    torch.Tensor
    """
    newg = deepcopy(g)
    # newg = newg.remove_self_loop().add_self_loop()
    # D^-1
    inv_degree = torch.pow(newg.out_degrees().float(), -1)
    inv_degree = torch.where(torch.isinf(inv_degree), torch.full_like(inv_degree, eps), inv_degree)
    inv_degree = torch.diag(inv_degree)
    # A
    adj = newg.adj().to_dense()
    mat = inv_degree @ adj 
    
    P_0 = torch.eye(newg.number_of_nodes()).float()
    X = torch.zeros_like(P_0) 
    P = P_0 
    for i in range(k): 
        P = r * P @ mat + (1 - r) * P_0 
        X += P 
    X /= k
    
    return X
    

def loss_func(x, x_hat, c, c_hat, h_a, h_s, hom_str, hom_attr, alphas, scale_factor, pretrain=False):
    """loss function

    Parameters
    ----------
    x : torch.Tensor
        adjacency matrix of the original graph
    x_hat : torch.Tensor
        adjacency matrix of the reconstructed graph
    c : torch.Tensor
        attribute matrix of the original graph
    c_hat : torch.Tensor
        attribute matrix of the reconstructed graph
    h_a : torch.Tensor
        embedding of attribute autoencoders
    h_s : torch.Tensor
        embedding of structure autoencoders
    hom_str : torch.Tensor
        intermediate value of homogeneity loss of structure autoencoder
    hom_attr : torch.Tensor
        intermediate value of homogeneity loss of attribute autoencoder
    alphas : list
        balance parameters
    scale_factor : float
        scale factor
    pretrain : bool, optional
        whether to pre-train, by default False

    Returns
    -------
    loss : torch.Tensor
        loss value
    score : torch.Tensor
        outlier score
    """
    # closed form update rules
    # Eq.8 struct score
    ds = torch.norm(x-x_hat, dim=1)
    numerator = alphas[0] * ds + alphas[1] * hom_str
    os = numerator / torch.sum(numerator) * scale_factor
    # Eq.9 attr score
    da = torch.norm(c-c_hat, dim=1)
    numerator = alphas[2] * da + alphas[3] * hom_attr
    oa = numerator / torch.sum(numerator) * scale_factor
    # Eq.10 com score
    dc = torch.norm(h_s-h_a, dim=1)
    oc = dc / torch.sum(dc) * scale_factor
    
    # using Adam
    if pretrain is True:
        loss_prox_str = torch.mean(ds)          # Eq.2
        loss_hom_str = torch.mean(hom_str)      # Eq.3
        loss_prox_attr = torch.mean(da)         # Eq.4
        loss_hom_attr = torch.mean(hom_attr)    # Eq.5
        loss_com = torch.mean(dc)               # Eq.6
    else:
        loss_prox_str = torch.mean(torch.log(torch.pow(os, -1)) * ds)       # Eq.2
        loss_hom_str = torch.mean(torch.log(torch.pow(os, -1)) * hom_str)   # Eq.3
        loss_prox_attr = torch.mean(torch.log(torch.pow(oa, -1)) * da)      # Eq.4
        loss_hom_attr = torch.mean(torch.log(torch.pow(oa, -1)) * hom_attr) # Eq.5
        loss_com = torch.mean(torch.log(torch.pow(oc, -1)) * dc)            # Eq.6
        
    # Eq.7
    loss = alphas[0] * loss_prox_str + alphas[1] * loss_hom_str + alphas[2] * loss_prox_attr + alphas[3] * loss_hom_attr + alphas[4] * loss_com
    
    score = (oa + os + oc) / 3
    return loss, score
  
    
def train_step(model, optimizer:torch.optim.Optimizer, g:dgl.DGLGraph, adj:torch.Tensor, batch_size:int, alphas:list, num_neighbors:int, device, pretrain=False):
    """train model in one epoch

    Parameters
    ----------
    model : class
        DONE base model
    optimizer : torch.optim.Optimizer
        optimizer to adjust model
    g : dgl.DGLGraph
        original graph
    adj : torch.Tensor
        adjacency matrix
    batch_size : int
        the size of training batch
    alphas : list
        balance parameters
    num_neighbors : int
        number of sampling neighbors
    device : str
        device of computation
    pretrain : bool, optional
        whether to pre-train, by default False

    Returns
    -------
    predict_score : numpy.ndarray
        outlier score
    epoch_loss : torch.Tensor
        loss value for epoch
    """
    # g = deepcopy(g)
    model.train()
    sampler = SubgraphNeighborSampler(num_neighbors)
    dataloader = dgl.dataloading.DataLoader(
        g, torch.arange(g.num_nodes()), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    epoch_loss = 0
    predict_score = torch.zeros(g.num_nodes())
    
    for sg in dataloader:
        feat = sg.ndata['feat']
        indices = sg.ndata['_ID']
        sub_adj = adj[indices]
        scale_factor = 1.0 * sg.num_nodes() / g.num_nodes()
        
        sg = sg.to(device)
        sub_adj = sub_adj.to(device)
        feat = feat.to(device)
        
        h_s, x_hat, h_a, c_hat, hom_str, hom_attr = model(sg, sub_adj, feat)
        loss, score = loss_func(sub_adj, x_hat, feat, c_hat, h_a, h_s, hom_str, hom_attr, alphas, scale_factor, pretrain=pretrain)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * sg.num_nodes()
        predict_score[indices] = score.detach().cpu()
        
    epoch_loss /= g.num_nodes()
        
    return predict_score, epoch_loss


def test_step(model, g, adj, batch_size, alphas, device):
    """test model in one epoch

    Parameters
    ----------
    model : nn.Module
        DONE base model 
    g : dgl.DGLGraph
        graph data
    adj : torch.Tensor
        adjacency matrix
    batch_size : int
        the size of training batch
    alphas : list
        balance parameters
    device : str
        device of computation

    Returns
    -------
    numpy.ndarray
    """
    # g = deepcopy(g)
    model.eval()
    sampler = SubgraphNeighborSampler()
    dataloader = dgl.dataloading.DataLoader(
        g, torch.arange(g.num_nodes()), sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    predict_score = torch.zeros(g.num_nodes())
    
    for sg in tqdm(dataloader):
        feat = sg.ndata['feat']
        indices = sg.ndata['_ID']
        sub_adj = adj[indices]
        scale_factor = 1.0 * sg.num_nodes() / g.num_nodes()
        
        sg = sg.to(device)
        sub_adj = sub_adj.to(device)
        feat = feat.to(device)
        
        h_s, x_hat, h_a, c_hat, hom_str, hom_attr = model(sg, sub_adj, feat)
        _, score = loss_func(sub_adj, x_hat, feat, c_hat, h_a, h_s, hom_str, hom_attr, alphas, scale_factor)

        predict_score[indices] = score.detach().cpu()
        
    return predict_score.numpy()


class SubgraphNeighborSampler(dgl.dataloading.Sampler):
    """the neighbor sampler of the subgraph

    Parameters
    ----------
    num_neighbors : int, optional
        number of sampling neighbors, by default -1
    """
    def __init__(self, num_neighbors=-1):
        super().__init__()
        self.num_neighbors = num_neighbors

    def sample(self, g, indices):
        g = g.subgraph(indices)
        g = dgl.sampling.sample_neighbors(g, g.nodes(), self.num_neighbors)
        return g