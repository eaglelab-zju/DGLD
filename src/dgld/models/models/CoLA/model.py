import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import EdgeWeightNorm, GraphConv, GATConv
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling

from .dataset import CoLADataSet
from .colautils import train_epoch, test_epoch

import numpy as np

class Discriminator(nn.Module):
    def __init__(self, out_feats):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(out_feats, out_feats, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, readout_emb, anchor_emb):
        logits = self.bilinear(readout_emb, anchor_emb)
        return logits


class OneLayerGCNWithGlobalAdg(nn.Module):
    r"""
    a onelayer subgraph GCN can use global adjacent metrix.
    """

    def __init__(self, in_feats, out_feats=64, global_adg=True):
        super(OneLayerGCNWithGlobalAdg, self).__init__()
        self.global_adg = global_adg
        self.norm = 'none' if global_adg else 'both'
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.conv = GraphConv(in_feats, out_feats,
                              weight=False, bias=False, norm=self.norm)
        self.conv.set_allow_zero_in_degree(1)
        self.act = nn.PReLU()
        self.reset_parameters()
        self.pool = AvgPooling()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, bg, in_feat, subgraph_size=4):
        anchor_embs = bg.ndata['feat'][::4, :].clone()
        # Anonymization
        bg.ndata['feat'][::4, :] = 0
        # anchor_out
        anchor_out = torch.matmul(anchor_embs, self.weight) + self.bias
        anchor_out = self.act(anchor_out)

        in_feat = bg.ndata['feat']
        in_feat = torch.matmul(in_feat, self.weight)
        # GCN
        if self.global_adg:
            h = self.conv(bg, in_feat, edge_weight=bg.edata['w'])
        else:
            h = self.conv(bg, in_feat)
        h += self.bias
        h = self.act(h)
        with bg.local_scope():
            # pooling
            subgraph_pool_emb = self.pool(bg, h)
        return F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1)


class OneLayerGCN(nn.Module):
    def __init__(self, in_feats=300, out_feats=64, bias=True):
        super(OneLayerGCN, self).__init__()
        self.conv = GraphConv(in_feats, out_feats, bias=bias)
        self.act = nn.PReLU()

    def forward(self, bg, in_feat):
        h = self.conv(bg, in_feat)
        h = self.act(h)
        with bg.local_scope():
            bg.ndata["h"] = h
            # subgraph_pool_emb = dgl.mean_nodes(bg, "h")
            subgraph_pool_emb = []
            # get anchor embedding
            unbatchg = dgl.unbatch(bg)
            anchor_out = []
            for g in unbatchg:
                subgraph_pool_emb.append(torch.mean(g.ndata["h"][:-1], dim=0))
                anchor_out.append(g.ndata["h"][-1])
            anchor_out = torch.stack(anchor_out, dim=0)
            subgraph_pool_emb = torch.stack(subgraph_pool_emb, dim=0)
        return subgraph_pool_emb, anchor_out
        # return F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1)


class CoLAModel(nn.Module):
    def __init__(self, in_feats=300, out_feats=64, global_adg=True):
        """[summary]

        Parameters
        ----------
        in_feats : int, optional
            [description], by default 300
        out_feats : int, optional
            [description], by default 64
        global_adg : bool, optional
            [description], by default True
        """        
        super(CoLAModel, self).__init__()
        self.gcn = OneLayerGCNWithGlobalAdg(in_feats, out_feats, global_adg)
        self.discriminator = Discriminator(out_feats)

    def forward(self, pos_batchg, pos_in_feat, neg_batchg, neg_in_feat):
        pos_pool_emb, anchor_out = self.gcn(pos_batchg, pos_in_feat)
        neg_pool_emb, _ = self.gcn(neg_batchg, neg_in_feat)
        pos_scores = self.discriminator(pos_pool_emb, anchor_out)
        neg_scores = self.discriminator(neg_pool_emb, anchor_out)
        return pos_scores[:, 0], neg_scores[:, 0]


class CoLA():    
    def __init__(self, in_feats=1433, out_feats=64, global_adg=True):
        """[CoLA]
        CoLA Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning 
        liu2021anomaly
        Parameters
        ----------
        in_feats : int, optional
            [dimension of input feat], by default 1433
        out_feats : int, optional
            [dimension of final embedding], by default 64
        global_adg : bool, optional
            [Whether to use the global adjacency matrix], by default True
        Examples:
        ---------
        ```python
        >>> from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
        >>> from DGLD.CoLA import CoLA
        >>> if __name__ == '__main__':
        >>>     # sklearn-like API for most users.
        >>>     gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
        >>>     g = gnd_dataset[0]
        >>>     model = CoLA(in_feats=1433)
        >>>     model.fit(g, num_epoch=1, device='cpu')
        >>>     result = model.predict(g, auc_test_rounds=4)
        >>>     gnd_dataset.evaluation_multiround(result) 
        ```
        """        
        self.model = CoLAModel(in_feats, out_feats, global_adg)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def fit(self, g, device='cpu', batch_size=300, lr=0.003, weight_decay=1e-5, num_workers=4, num_epoch=100, logdir='tmp', seed=42):
        """[summary]
        Parameters
        ----------
        g : [dgl.Graph]
            [input graph with feature named "feat" in g.ndata.]
        device : str, optional
            [device], by default 'cpu'
        batch_size : int, optional
            [batch size for training], by default 300
        lr : float, optional
            [learning rate for training], by default 0.003
        weight_decay : [float], optional
            [weight decay for training], by default 1e-5
        num_workers : int, optional
            [num_workers using in `pytorch DataLoader`], by default 4
        num_epoch : int, optional
            [number of epoch for training], by default 100
        logdir : str, optional
            [tensorboard logdir], by default 'tmp'
        Returns
        -------
        [self]
            return the model self.
        """        
        dataset = CoLADataSet(g)
        train_loader = GraphDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            shuffle=True,
        )

        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + str(device))
        else:
            device = torch.device("cpu")
        self.model.to(device)

        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        writer = SummaryWriter(log_dir=logdir)

        for epoch in range(num_epoch):
            train_loader.dataset.random_walk_sampling()
            loss_accum = train_epoch(
                epoch, train_loader, self.model, device, self.criterion, optimizer
            )
            writer.add_scalar("loss", float(loss_accum), epoch)
        return self

    def predict(self, g, device='cpu', batch_size=300, num_workers=4, auc_test_rounds=256, logdir='tmp'):
        """[summary]

        Parameters
        ----------
        g : [type]
            [description]
        device : str, optional
            [description], by default 'cpu'
        batch_size : int, optional
            [description], by default 300
        num_workers : int, optional
            [description], by default 4
        auc_test_rounds : int, optional
            [description], by default 256
        logdir : str, optional
            [description], by default 'tmp'
        Returns
        -------
        [type]
            [description]
        """
        dataset = CoLADataSet(g)
        test_loader = GraphDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
        )
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + str(device))
        else:
            device = torch.device("cpu")
        self.model.to(device)

        predict_score_arr = []
        for rnd in range(auc_test_rounds):
            test_loader.dataset.random_walk_sampling()
            predict_score = test_epoch(
                rnd, test_loader, self.model, device, self.criterion
            )
            predict_score_arr.append(list(predict_score))
        predict_score_arr = np.array(predict_score_arr).T
        return predict_score_arr.mean(1)
