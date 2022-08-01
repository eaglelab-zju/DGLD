# -*- coding: utf-8 -*-
import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from  dgl.nn.pytorch import EdgeWeightNorm
from dgl.dataloading import GraphDataLoader
from .dataset import CoLADataSet
from .anemone_utils import train_epoch, test_epoch
import torch.optim as optim
import numpy as np
from utils.early_stopping import EarlyStopping

class Discriminator(nn.Module):
    """
    This is a discriminator component for contrastive learning of positive subgraph and negative subgraph
    Parameters
    ----------
    out_feats : int
        The number of class to distinguish
    """
    def __init__(self, out_feats):
        super(Discriminator, self).__init__()
        self.bilinear_1 = nn.Bilinear(out_feats, out_feats, 1)
        self.bilinear_2= nn.Bilinear(out_feats, out_feats, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """
        Functions that init weights of discriminator component
        Parameters
        ----------
        m : nn.Parameter
            the parameter to initial

        Returns
        -------
        None
        """
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, readout_emb,rec_emb, anchor_emb_1,anchor_emb_2):
        """
        Functions that compute bilinear of subgraph embedding and node embedding
        Parameters
        ----------
        readout_emb : Torch.tensor
            the subgraph embedding
        rec_emb : Torch.tensor
            the recovery of target node
        anchor_emb_1 : Totch.tensor
            the node embedding
        anchor_emb_2 : Totch.tensor
            the node embedding
        Returns
        -------
        logits : Torch.tensor
            the logit after bilinear
        """
        logits_rdt = (self.bilinear_1(readout_emb, anchor_emb_1))
        logits_rec=(self.bilinear_2(rec_emb, anchor_emb_2))
        return logits_rdt,logits_rec


class OneLayerGCNWithGlobalAdg(nn.Module):
    """
    a onelayer subgraph GCN can use global adjacent metrix.
    Parameters
    ----------
    in_feats : Torch.tensor
        the feature dimensions of input data
    out_feats : Torch.tensor, optional
        the feature dimensions of output data, default 64
    global_adg : bool, optional
        whether use the global information of node, here means the degree matrix, default True
    """
    def __init__(self, in_feats, out_feats=64, global_adg=True
                 ):
        super(OneLayerGCNWithGlobalAdg, self).__init__()
        self.global_adg = global_adg
        self.norm = 'both'
        self.weight1 = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight2 = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias1 = nn.Parameter(torch.Tensor(out_feats))
        self.bias2 = nn.Parameter(torch.Tensor(out_feats))
        self.conv1 = GraphConv(in_feats, out_feats, weight=False, bias=False, norm=self.norm)
        self.conv1.set_allow_zero_in_degree(1)
        self.conv2 = GraphConv(in_feats, out_feats, weight=False, bias=False, norm=self.norm)
        self.conv2.set_allow_zero_in_degree(1)
        self.act = nn.PReLU()
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        if self.weight1 is not None:
            init.xavier_uniform_(self.weight1)
        if self.weight2 is not None:
            init.xavier_uniform_(self.weight2)
        if self.bias1 is not None:
            init.zeros_(self.bias1)
        if self.bias2 is not None:
            init.zeros_(self.bias2)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, bg, in_feat):
        """
        The function to compute forward of GCN
        Parameters
        ----------
        bg : list of dgl.heterograph.DGLHeteroGraph
            the list of subgraph, to compute forward and loss
        in_feat : Torch.tensor
            the node feature of geive subgraph
        Returns
        -------
        h : Torch.tensor
            the embedding of batch subgraph node after one layer GCN
        subgraph_pool_emb : Torch.tensor
            the embedding of batch subgraph after one layer GCN, aggregation of batch subgraph node embedding
        subgraph_rec_emb : Torch.tensor
            the recovery embedding of target node
        anchor_out_1 : Torch.tensor
            the embedding of batch anchor node
        anchor_out_1 : Torch.tensor
            the embedding of batch anchor node
        """
        # Anonymization
        unbatchg = dgl.unbatch(bg)
        unbatchg_list = []
        anchor_feat_list = []
        for g in unbatchg:
            anchor_feat = g.ndata['feat'][0, :].clone()
            g.ndata['feat'][0, :] = 0
            unbatchg_list.append(g)
            anchor_feat_list.append(anchor_feat)

        # anchor_out
        anchor_embs = torch.stack(anchor_feat_list, dim=0)
        anchor_out_1 = torch.matmul(anchor_embs, self.weight1)
        anchor_out_1=anchor_out_1+self.bias1
        anchor_out_1=self.act(anchor_out_1)
        anchor_out_2 = torch.matmul(anchor_embs, self.weight2)
        anchor_out_2=anchor_out_2+self.bias2
        anchor_out_2 = self.act(anchor_out_2)
        bg = dgl.batch(unbatchg_list)
        in_feat = bg.ndata['feat']
        # in_feat_1 = torch.matmul(in_feat, self.weight1)
        # in_feat_2 = torch.matmul(in_feat, self.weight2)
        # GCN
        if self.global_adg:
            h1 = self.conv1(bg, in_feat, edge_weight=bg.edata['w'])
            h2 = self.conv2(bg, in_feat, edge_weight=bg.edata['w'])
        else:
            h1 = self.conv1(bg, in_feat)
            h2 = self.conv2(bg, in_feat)
        # h1 += self.bias1
        h1=torch.matmul(h1, self.weight1)
        h2=torch.matmul(h2, self.weight2)
        h1=h1+self.bias1
        h2=h2+self.bias2
        h1 = self.act(h1)
        # h2 += self.bias2
        h2 = self.act(h2)
        with bg.local_scope():
            # pooling
            bg.ndata["h1"] = h1
            bg.ndata["h2"]=h2
            subgraph_pool_emb = []
            subgraph_rec_emb=[]
            unbatchg = dgl.unbatch(bg)
            for g in unbatchg:
                tmp_data=g.ndata["h1"][1:,:]
                # subgraph_pool_emb.append(torch.mean(g.ndata["h1"], dim=0))
                subgraph_pool_emb.append(torch.mean(tmp_data, dim=0))
                subgraph_rec_emb.append(g.ndata["h2"][0])
            subgraph_pool_emb = torch.stack(subgraph_pool_emb, dim=0)
            subgraph_rec_emb=torch.stack(subgraph_rec_emb, dim=0)
        # return subgraph_pool_emb, subgraph_rec_emb,anchor_out_1,anchor_out_2
        return F.normalize(subgraph_pool_emb, p=2, dim=1),F.normalize(subgraph_rec_emb, p=2, dim=1),F.normalize(anchor_out_1, p=2, dim=1),F.normalize(anchor_out_2, p=2, dim=1)





class AneModel(nn.Module):
    def __init__(self, in_feats=300, out_feats=64, global_adg=False):
        """AneModel
        Parameters
        ----------
        in_feats : int, optional
            the feature dimensions of input data, by default 300
        out_feats : int, optional
            the feature dimensions of output data, by default 64
        global_adg : bool, optional
            whether use the global information of node, default True
        """
        super(AneModel, self).__init__()
        self.gcn = OneLayerGCNWithGlobalAdg(in_feats, out_feats, global_adg)
        # self.gcn2 = OneLayerGCNWithGlobalAdg(in_feats, out_feats, global_adg)
        self.discriminator = Discriminator(out_feats)

    def forward(self, pos_batchg, pos_in_feat, neg_batchg, neg_in_feat):
        """
        The function to compute forward and loss of SL-GAD model
        Parameters
        ----------
        pos_batchg : DGL.Graph
            batch of positive subgraph
        pos_in_feat : Torch.tensor
            node features of positive subgraph batch
        neg_batchg : DGL.Graph
            batch of negative subgraph
        neg_in_feat : Torch.tensor
            node features of negative subgraph batch
        Returns
        -------
        pos_scores_rdt : Torch.tensor
            anomaly score of positive sample
        pos_scores_rec : Torch.tensor
            anomaly score of positive sample
        neg_scores_rdt : Torch.tensor
            anomaly score of negative sample
        neg_scores_rec : Torch.tensor
            anomaly score of negative sample
        """
        pos_pool_emb,pos_rec_emb, anchor_out_pos_1,anchor_out_pos_2 = self.gcn(pos_batchg, pos_in_feat)
        neg_pool_emb,neg_rec_emb, _,_ = self.gcn(neg_batchg, neg_in_feat)
        pos_scores_rdt,pos_scores_rec = self.discriminator(pos_pool_emb,pos_rec_emb, anchor_out_pos_1,anchor_out_pos_2)
        # neg_pool_emb=pos_pool_emb,
        # neg_pool_emb=torch.cat((pos_pool_emb[-2:-1,:],pos_pool_emb[:-1,:]),0)
        # neg_rec_emb=pos_rec_emb
        # neg_rec_emb = torch.cat((pos_rec_emb[-2:-1,:], pos_rec_emb[:-1,:]), 0)
        neg_scores_rdt,neg_scores_rec = self.discriminator(neg_pool_emb,neg_rec_emb, anchor_out_pos_1,anchor_out_pos_2)
        return pos_scores_rdt[:, 0], pos_scores_rec[:, 0], neg_scores_rdt[:, 0],neg_scores_rec[:, 0]


class ANEMONE():
    def __init__(self, in_feats=1433, out_feats=64, global_adg=True):
        """
        CoLA Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning
        liu2021anomaly

        Parameters
        ----------
        in_feats : int, optional
            dimension of input feat, by default 1433
        out_feats : int, optional
            dimension of final embedding, by default 64
        global_adg : bool, optional
            Whether to use the global adjacency matrix, by default True

        Examples
        -------
        >>> from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
        >>> from DGLD.ANEMONE import ANEMONE
        >>> if __name__ == '__main__':
        >>>     # sklearn-like API for most users.
        >>>     gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
        >>>     g = gnd_dataset[0]
        >>>     model = CoLA(in_feats=1433)
        >>>     model.fit(g, num_epoch=1, device='cpu')
        >>>     result = model.predict(g, auc_test_rounds=4)
        >>>     gnd_dataset.evaluation_multiround(result)
        """
        self.model = AneModel(in_feats, out_feats, global_adg)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def fit(self, g, device='cpu', batch_size=300, lr=0.003, weight_decay=1e-5, num_workers=4, num_epoch=100,
            seed=42,alpha=0.8):
        """train the model

        Parameters
        ----------
        g : dgl.Graph
            input graph with feature named "feat" in g.ndata.
        device : str, optional
            device, by default 'cpu'
        batch_size : int, optional
            batch size for training, by default 300
        lr : float, optional
            learning rate for training, by default 0.003
        weight_decay : float, optional
            weight decay for training, by default 1e-5
        num_workers : int, optional
            num_workers using in `pytorch DataLoader`, by default 4
        num_epoch : int, optional
            number of epoch for training, by default 100


        Returns
        -------
        self : model
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
        early_stop = EarlyStopping(early_stopping_rounds=10, patience=10)
        for epoch in range(num_epoch):
            train_loader.dataset.random_walk_sampling()
            loss_accum = train_epoch(
                epoch,alpha, train_loader, self.model, device, self.criterion, optimizer
            )
            early_stop(loss_accum, self.model)
            if early_stop.isEarlyStopping():
                print(f"Early stopping in round {epoch}")
                break
        return self

    def predict(self, g, device='cpu', batch_size=300, num_workers=4, auc_test_rounds=256, alpha=0.8):
        """test model

        Parameters
        ----------
        g : type
            description
        device : str, optional
            description, by default 'cpu'
        batch_size : int, optional
            description, by default 300
        num_workers : int, optional
            description, by default 4
        auc_test_rounds : int, optional
            description, by default 256

        Returns
        -------
        predict_score_arr : numpy.ndarray
            description
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
                rnd, alpha,test_loader, self.model, device, self.criterion
            )
            predict_score_arr.append(list(predict_score))
        predict_score_arr = np.array(predict_score_arr).T
        return predict_score_arr.mean(1)

