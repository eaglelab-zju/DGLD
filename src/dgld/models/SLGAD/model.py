import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from  dgl.nn.pytorch import EdgeWeightNorm
import math
import numpy
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
from sklearn.preprocessing import MinMaxScaler

import scipy.sparse as sp
import numpy as np

# torch.set_printoptions(precision=8)
# torch.set_default_tensor_type(torch.DoubleTensor)

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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
        self.bilinear = nn.Bilinear(out_feats, out_feats, 1)
        # print(self.bilinear.weight)
        for m in self.modules():
            self.weights_init(m)
        # print(self.bilinear.weight)
        torch.nn.init.constant_(self.bilinear.weight, 0.5)
        torch.nn.init.constant_(self.bilinear.bias, 0.0)
        # exit()

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

    def forward(self, readout_emb, anchor_emb):
        """
        Functions that compute bilinear of subgraph embedding and node embedding

        Parameters
        ----------
        readout_emb : Torch.tensor
            the subgraph embedding
        anchor_emb : Totch.tensor
            the node embedding

        Returns
        -------
        logits : Torch.tensor
            the logit after bilinear
        """
        logits = self.bilinear(anchor_emb, readout_emb)
        return logits

class OneLayerGCNWithGlobalAdg_simple(nn.Module):
    """
    A onelayer subgraph GCN can use global adjacent metrix.

    Parameters
    ----------
    in_feats : Torch.tensor
        the feature dimensions of input data
    out_feats : Torch.tensor, optional
        the feature dimensions of output data, default 64
    global_adg : bool, optional
        whether use the global information of node, here means the degree matrix, default True
    """
    def __init__(self, in_feats, out_feats=64, global_adg=True):
        super(OneLayerGCNWithGlobalAdg_simple, self).__init__()
        self.global_adg = global_adg
        self.norm = 'none' if global_adg else 'both'
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.conv = GraphConv(in_feats, out_feats, weight=False, bias=False, norm=self.norm)
        self.conv.set_allow_zero_in_degree(1)
        self.act = nn.PReLU()
        self.reset_parameters()
        self.pool = AvgPooling()

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, bg, in_feat, subgraph_size=4):
        """
        The function to compute forward of GCN

        Parameters
        ----------
        bg : list of dgl.heterograph.DGLHeteroGraph
            the list of subgraph, to compute forward and loss
        in_feat : Torch.tensor
            the node feature of geive subgraph
        anchor_embs : Torch.tensor
            the anchor embeddings
        attention : Functions, optional
            attention machanism, default None

        Returns
        -------
        h : Torch.tensor
            the embedding of batch subgraph node after one layer GCN
        subgraph_pool_emb : Torch.tensor
            the embedding of batch subgraph after one layer GCN, aggregation of batch subgraph node embedding
        anchor_out : Torch.tensor
            the embedding of batch anchor node
        """
        anchor_embs = in_feat[::4, :].clone()
        # Anonymization
        in_feat[::4, :] = 0
        # anchor_out
        anchor_out = torch.matmul(anchor_embs, self.weight) + self.bias
        anchor_out = self.act(anchor_out)

        # in_feat = bg.ndata['feat']
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

class OneLayerGCNWithGlobalAdg(nn.Module):
    """
    A onelayer subgraph GCN can use global adjacent metrix.

    Parameters
    ----------
    in_feats : Torch.tensor
        the feature dimensions of input data
    out_feats : Torch.tensor, optional
        the feature dimensions of output data, default 64
    global_adg : bool, optional
        whether use the global information of node, here means the degree matrix, default True
    args : parser, optional
        extra custom made of model, default None
    """
    def __init__(self, in_feats, out_feats=64, global_adg=True, args = None):
        super(OneLayerGCNWithGlobalAdg, self).__init__()
        self.global_adg = global_adg
        self.norm = 'none' if global_adg else 'both'
        # self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight = nn.Linear(in_feats, out_feats, bias=False)
        # self.weight.double()
        # self.weight.data.fill_(0.0)
        # print(self.weight.weight.dtype)
        # exit()
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.bias.data.fill_(0.0)

        for m in self.modules():
            self.weights_init(m)
        # print(self.weight.weight)
        # exit()
        self.conv = GraphConv(in_feats, out_feats, weight=False, bias=False, norm=self.norm)
        self.conv.set_allow_zero_in_degree(1)
        # print(args)
        # print(args.act_function)
        if args == None or args.act_function == "PReLU":
            self.act = nn.PReLU()
        elif args.act_function == "ReLU":
            self.act = nn.ReLU()
        # self.reset_parameters()
        self.pool = AvgPooling()
        if args == None:
            pass
            self.attention = None
        else:
            pass
            self.attention = args.attention
        # print(self.weight.weight)
        # exit()
        
    def weights_init(self, m):
        """
        Init the weight of Linear

        Parameters
        ----------
        m : nn.model
            the model to transform weight of linear
        
        Returns
        -------
        None
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, bg, in_feat, anchor_embs, attention = None):
        """
        The function to compute forward of GCN

        Parameters
        ----------
        bg : list of dgl.heterograph.DGLHeteroGraph
            the list of subgraph, to compute forward and loss
        in_feat : Torch.tensor
            the node feature of geive subgraph
        anchor_embs : Torch.tensor
            the anchor embeddings
        attention : Functions, optional
            attention machanism, default None

        Returns
        -------
        h : Torch.tensor
            the embedding of batch subgraph node after one layer GCN
        subgraph_pool_emb : Torch.tensor
            the embedding of batch subgraph after one layer GCN, aggregation of batch subgraph node embedding
        anchor_out : Torch.tensor
            the embedding of batch anchor node
        """
        # print(bg.edata['w'][:20])
        # print(bg.edges()[0][:20])
        # print(bg.edges()[1][:20])
        temp_weight = bg.edata['w'][:20].cpu().numpy()
        temp_src = bg.edges()[0][:20].cpu().numpy()
        temp_dst = bg.edges()[1][:20].cpu().numpy()
        max_num = max([np.max(temp_src), np.max(temp_dst)])
        temp_matrix = sp.coo_matrix((temp_weight, (temp_src, temp_dst)), shape = (np.max(temp_src) + 1, np.max(temp_dst) + 1))
        # print(temp_matrix.todense()[:4, :4])
        # exit()
        # anchor_out = torch.matmul(anchor_embs, self.weight) + self.bias
        # print(anchor_embs.dtype)
        anchor_out = self.weight(anchor_embs) + self.bias
        anchor_out = self.act(anchor_out)
        # bg = dgl.batch(unbatchg_list)
        
        # in_feat = bg.ndata['feat']
        # h = torch.matmul(in_feat, self.weight)
        # print('before linear : ', in_feat[:5, :5])
        h = self.weight(in_feat)
        # print(torch.nonzero(in_feat[:5, :]))
        # print('in GCN : ')
        # print(self.weight.weight[:5, :5])
        # print(in_feat[:5, :5])
        # print(h[:5, :5])
        # exit()
        # GCN
        # print(bg)
        # print(bg.edges()[0][:50])
        # print(bg.edges()[1][:50])
        # print(bg.edata['w'][:50])
        # print(bg.in_degrees(bg.nodes())[:50])
        # exit()
        # print(h[:5, :5])
        # print(bg.edata['w'][:20])
        # print(bg.edges())
        # print(bg.edges()[0][:20])
        # print(bg.edges()[1][:20])
        # print('before con : ', h[:5, :5])
        if self.global_adg:
            h = self.conv(bg, h, edge_weight=bg.edata['w'])
        else:
            h = self.conv(bg, h)
        h += self.bias
        # print('before act : ', h[:5, :5])
        h = self.act(h)
        with bg.local_scope():
            subgraph_pool_emb = self.pool(bg, h)
            # Time_Process.global_time.process_Time("after GCN bg unbatchg")
        # print(in_feat.shape)
        # print(in_feat[:5, :5])
        # print(h[:5, :5])
        # exit()
        return h, subgraph_pool_emb, anchor_out
        return F.normalize(h, p=2, dim=1), F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1)


class OneLayerGCN(nn.Module):
    """
    A onelayer subgraph GCN can use global adjacent metrix.

    Parameters
    ----------
    in_feats : Torch.tensor, optional
        the feature dimensions of input data, default 300
    out_feats : Torch.tensor, optional
        the feature dimensions of output data, default 64
    bias : bool, optional
        whether the bias of model exists or not, default True
    args : parser, optional
        extra custom made of model, default None
    """
    def __init__(self, in_feats=300, out_feats=64, bias=True, args = None):
        super(OneLayerGCN, self).__init__()
        self.conv = GraphConv(in_feats, out_feats, bias=bias)
        self.global_adg = args.global_adg
        if args == None or args.act_function == "PReLU":
            self.act = nn.PReLU()
        elif args.act_function == "ReLU":
            self.act = nn.ReLU()
        self.conv.set_allow_zero_in_degree(1)

    def forward(self, bg, in_feat):
        """
        The function to compute forward and loss of model with given subgraph

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
        anchor_out : Torch.tensor
            the embedding of batch anchor node
        """
        if self.global_adg:
            h = self.conv(bg, in_feat, edge_weight=bg.edata['w'])
        else:
            h = self.conv(bg, in_feat)
        
        h = self.act(h)
        with bg.local_scope():
            bg.ndata["h"] = h
            bg.ndata["in_feat"] = in_feat
            # subgraph_pool_emb = dgl.mean_nodes(bg, "h")
            subgraph_pool_emb = []
            # get anchor embedding
            unbatchg = dgl.unbatch(bg)
            anchor_out = []
            for g in unbatchg:
                # embedding = enc_gat(g, g.ndata["h"])
                # subgraph_pool_emb.append(embedding)
                subgraph_pool_emb.append(torch.mean(g.ndata["h"], dim=0))
                single_anchor_out = torch.matmul(g.ndata["in_feat"][0], self.conv.weight) + self.conv.bias
                single_anchor_out = self.act(single_anchor_out)
                anchor_out.append(single_anchor_out)
                # anchor_out.append(g.ndata["h"][0])
            anchor_out = torch.stack(anchor_out, dim=0)
            subgraph_pool_emb = torch.stack(subgraph_pool_emb, dim=0)
        return F.normalize(h, p = 2, dim = 1), F.normalize(subgraph_pool_emb, p = 2, dim = 1), F.normalize(anchor_out, p = 2, dim = 1)
        # return F.normalize(subgraph_pool_emb, p=2, dim=1), F.normalize(anchor_out, p=2, dim=1)

import time
# import Time_Process
class SL_GAD_Model(nn.Module):
    """
    SL-GAD_model, given two positive subgraph and one negative subgraph, return the loss and score of target nodes

    Parameters
    ----------
    in_feats : Torch.tensor, optional
        the feature dimensions of input data, default 300
    out_feats : Torch.tensor, optional
        the feature dimensions of output data, default 64
    global_adg : bool, optional
        whether use the global information of node, here means the degree matrix, default True
    args : parser, optional
        extra custom made of model, default None
    """
    def __init__(self, in_feats=300, out_feats=64, global_adg=True, args = None):
        super(SL_GAD_Model, self).__init__()
        self.enc = OneLayerGCNWithGlobalAdg(in_feats, out_feats, global_adg, args)
        self.dec = OneLayerGCNWithGlobalAdg(out_feats, in_feats, global_adg, args)
        self.enc_simple = OneLayerGCNWithGlobalAdg_simple(in_feats, out_feats, global_adg)
        # self.enc_gat = GATConv(out_feats, out_feats, num_heads=1, allow_zero_in_degree=True)
        self.attention = torch.nn.MultiheadAttention(embed_dim=out_feats, num_heads=1)
        self.discriminator_1 = Discriminator(out_feats)
        self.discriminator_2 = Discriminator(out_feats)
        self.args = args
        if args == None:
            self.alpha = 0.6
            self.beta = 0.4
        else:
            self.alpha = args.alpha
            self.beta = args.beta
        if args == None:
            self.device = 'cuda:' + '1'
        else:
            self.device = 'cuda:' + str(args.device)
        self.b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1]).to(self.device))
        # print('cuda:' + str(args.device))
        print('alpha : ', self.alpha)
        print('beta : ', self.beta)
        # exit()
        # print('in_feats', in_feats)
        # print('out_feats', out_feats)


    def forward(self, pos_batchg, pos_in_feat, neg_batchg, neg_in_feat, args):
        """
        The function to compute forward and loss of SL-GAD model

        Parameters
        ----------
        pos_batchg : list of dgl.heterograph.DGLHeteroGraph
            two batch of positive subgraph
        pos_in_feat : list of Torch.tensor
            node features of two batch of positive subgraph
        neg_batchg : list of dgl.heterograph.DGLHeteroGraph
            one batch of negative subgraph
        neg_in_feat : list of Torch.tensor
            node features of one batch of negative subgraph
        args : parser
            extra custom made of model

        Returns:
        ----------
        L : Torch.tensor
            loss of model
        single_predict_scores : Torch.tensor
            anomaly score of anchor nodes
        """
        if False:
            # temp_feat = pos_in_feat[0].clone()
            # pos_pool_emb, anchor_out = self.enc_simple(pos_batchg[0], temp_feat)
            # print(pos_pool_emb[0, :5])
            # print(anchor_out[0, :5])

            temp_feat = pos_in_feat[0].clone()
            temp_anchor = temp_feat[::4, :].clone()
            temp_feat[::4, :] = 0.0
            feat_1, pos_pool_emb, anchor_out = self.enc(pos_batchg[0], temp_feat, temp_anchor)
            # print(pos_pool_emb[0, :5])
            # print(anchor_out[0, :5])
            # exit()

            # neg_pool_emb, _ = self.enc_simple(neg_batchg, neg_in_feat)
            # print(pos_pool_emb.shape)
            # print(pos_pool_emb[:5])
            # c_mi = pos_pool_emb.clone()
            neg_pool_emb = torch.cat((pos_pool_emb[-1, :].unsqueeze(0), pos_pool_emb[:-1, :]), dim = 0)
            # print(neg_pool_emb[:5])
            # exit()
            pos_scores = self.discriminator(pos_pool_emb, anchor_out)
            neg_scores = self.discriminator(neg_pool_emb, anchor_out)
            pos_scores = torch.sigmoid(pos_scores)
            neg_scores = torch.sigmoid(neg_scores)
            loss = - torch.mean(torch.log(pos_scores[:, 0]) + torch.log(1 - neg_scores[:, 0])) / 2
            score = neg_scores[:, 0] - pos_scores[:, 0]
            return loss, score

        # print("before net preprocess", time.time())
        # Time_Process.global_time.process_Time("before net preprocess")
        # print(pos_batchg[0].edata['w'][:10])
        # exit()
        pos_batchg_1 = pos_batchg[0]
        pos_batchg_2 = pos_batchg[1]
        pos_in_feat_1 = pos_in_feat[0].clone()
        pos_in_feat_2 = pos_in_feat[1].clone()
        raw_pos_in_feat_1 = pos_in_feat[2].clone()
        raw_pos_in_feat_2 = pos_in_feat[3].clone()
        anchor_embs = pos_in_feat_1[::4, :].clone()
        anchor_embs_copy = anchor_embs.clone()
        raw_anchor_embs = raw_pos_in_feat_1[::4, :].clone()
        anchor_out_1 = pos_in_feat_1[::4, :].clone()
        anchor_out_2 = pos_in_feat_2[::4, :].clone()
        raw_anchor_out_1 = raw_pos_in_feat_1[::4, :].clone()
        raw_anchor_out_2 = raw_pos_in_feat_2[::4, :].clone()
        anchor_out_neg = neg_in_feat[::4, :].clone()
        pos_in_feat_1[::4, :] = 0.0
        pos_in_feat_2[::4, :] = 0.0
        raw_pos_in_feat_1[::4, :] = 0.0
        raw_pos_in_feat_2[::4, :] = 0.0
        
        feat_1, pos_pool_emb_1, anchor_out_1 = self.enc(pos_batchg_1, pos_in_feat_1, anchor_out_1)
        # print(pos_in_feat_1[:5, :5])
        # print(feat_1[:5, :5])
        # exit()
        # print(pos_in_feat_1[:5, :5])
        # print(feat_1[:5, :5])
        # print(pos_pool_emb_1[0, :5])
        # print(anchor_out_1[0, :5])
        # exit()
        feat_2, pos_pool_emb_2, anchor_out_2 = self.enc(pos_batchg_2, pos_in_feat_2, anchor_out_2)
        # feat_neg, neg_pool_emb, anchor_out_neg = self.enc(neg_batchg, neg_in_feat, anchor_out_neg)
        # print(pos_in_feat_1[:5])
        # print(pos_in_feat_2[:5])
        # print(anchor_out_1[:5])
        # print(anchor_out_2[:5])
        # exit()
        # feat_3, pos_pool_emb_3, anchor_out_3 = self.dec(pos_batchg_1, feat_1, anchor_out_1)
        # feat_4, pos_pool_emb_4, anchor_out_4 = self.dec(pos_batchg_2, feat_2, anchor_out_2)

        
        raw_feat_1, raw_pos_pool_emb_1, raw_anchor_out_1 = self.enc(pos_batchg_1, raw_pos_in_feat_1, raw_anchor_out_1)
        # print('done')
        # exit()
        raw_feat_2, raw_pos_pool_emb_2, raw_anchor_out_2 = self.enc(pos_batchg_2, raw_pos_in_feat_2, raw_anchor_out_2)
        raw_feat_3, raw_pos_pool_emb_3, raw_anchor_out_3 = self.dec(pos_batchg_1, raw_feat_1, raw_anchor_out_1)
        raw_feat_4, raw_pos_pool_emb_4, raw_anchor_out_4 = self.dec(pos_batchg_2, raw_feat_2, raw_anchor_out_2)
        # exit()
        # print(pos_in_feat_1[:5, :5])
        # print(feat_1[:5, :5])
        # print(pos_pool_emb_1[0, :5])
        # print(anchor_out_1[0, :5])
        # print(raw_feat_1[:5, :5])
        # print(self.dec.weight.weight[:5, :5])
        # print(raw_feat_3[:5, :5])
        # print(raw_feat_3.dtype)
        # print(self.dec.weight.weight[:5, :5])
        # exit()

        pos_scores_1 = self.discriminator_1(pos_pool_emb_1, anchor_out_2)
        pos_scores_2 = self.discriminator_2(pos_pool_emb_2, anchor_out_1)
        # print(feat_1[:20, :5])
        # print(pos_pool_emb_1[:10, :5])
        # print(anchor_out_2[:10, :5])
        # print(pos_scores_1[:10])
        # print(pos_scores_2[:10])
        # exit()

        neg_1 = pos_pool_emb_1
        neg_2 = pos_pool_emb_2
        neg_1 = torch.cat((neg_1[-1, :].unsqueeze(0), neg_1[:-1, :]), dim = 0)
        neg_2 = torch.cat((neg_2[-1, :].unsqueeze(0), neg_2[:-1, :]), dim = 0)

        neg_scores_1 = self.discriminator_1(neg_1, anchor_out_2)
        neg_scores_2 = self.discriminator_2(neg_2, anchor_out_1)
        
        # generative_diff_1 = anchor_out_3 - anchor_embs
        # generative_diff_2 = anchor_out_4 - anchor_embs

        # generative_diff_1 = raw_anchor_out_3 - raw_anchor_embs
        # generative_diff_2 = raw_anchor_out_4 - raw_anchor_embs
        
        # print(raw_feat_3[::4, :][:5, :5])
        # exit()
        generative_diff_1 = raw_feat_3[::4, :].clone() - raw_anchor_embs
        generative_diff_2 = raw_feat_4[::4, :].clone() - raw_anchor_embs
        # print(raw_feat_1[:5, :5])
        # print(raw_feat_3[::4, :][:5, :5])
        # print(raw_anchor_embs[:5, :5])
        # print(raw_feat_3.dtype)
        # exit()

        pos_scores_3 = (pos_scores_1 + pos_scores_2) / 2
        pos_scores_3 = torch.sigmoid(pos_scores_3)

        neg_scores_3 = (neg_scores_1 + neg_scores_2) / 2
        neg_scores_3 = torch.sigmoid(neg_scores_3)
        
        # print(pos_scores_3[:20])
        # print(neg_scores_3[:20])
        # exit()

        score_tot = torch.cat((torch.cat((pos_scores_1, pos_scores_2), dim = 1), torch.cat((neg_scores_1, neg_scores_2), dim = 1)), dim = 0)
        score_tot = torch.mean(score_tot, dim = 1, keepdim = True)
        lbl = torch.unsqueeze(torch.cat((torch.ones(anchor_embs.shape[0]),
                                                 torch.zeros(anchor_embs.shape[0]))), dim = 1).to(self.device)
        # loss_all = self.b_xent(score_tot, lbl)
        # print(loss_all[:10])
        # loss_all = torch.mean(loss_all)
        # print(loss_all)

        # loss_all = - torch.log((torch.sigmoid((pos_scores_1 + pos_scores_2) / 2)))
        # print(loss_all[:10])
        # loss_all = torch.mean(loss_all)
        # print(loss_all)
        # exit()
        # print(score_tot[:10])
        pos_scores_1 = torch.sigmoid(pos_scores_1)
        pos_scores_2 = torch.sigmoid(pos_scores_2)
        neg_scores_1 = torch.sigmoid(neg_scores_1)
        neg_scores_2 = torch.sigmoid(neg_scores_2)
        
        # print(score_tot.shape)
        # exit()
        # loss = - torch.mean(torch.log(pos_scores_1[:, 0]) + torch.log(1 - neg_scores_1[:, 0])) / 2
        # score = neg_scores_1[:, 0] - pos_scores_1[:, 0]
        # return loss, score

        contrastive_loss_1 = - torch.mean(torch.log(pos_scores_1 + 1e-8) + torch.log(1 - neg_scores_1 + 1e-8)) / 2
        contrastive_loss_2 = - torch.mean(torch.log(pos_scores_2 + 1e-8) + torch.log(1 - neg_scores_2 + 1e-8)) / 2
        contrastive_loss = (contrastive_loss_1 + contrastive_loss_2) / 2
        
        lbl = torch.unsqueeze(torch.cat((torch.ones(anchor_embs.shape[0]),
                                                 torch.zeros(anchor_embs.shape[0]))), dim = 1).to(self.device)
        # print(self.b_xent.device)
        # print(score_tot.to(self.device).device)
        # print(lbl.to(self.device).device)
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1]).to(self.device))
        loss_all = b_xent(score_tot.to(self.device), lbl.to(self.device))
        # print(loss_all[:10])
        loss_all = torch.mean(loss_all)
        # print(loss_all)

        # loss_all = - (torch.log((pos_scores_1 + pos_scores_2) / 2) + torch.log((2 - neg_scores_1 + neg_scores_2) / 2))
        # print(loss_all[:10])
        # loss_all = torch.mean(loss_all)
        # print(loss_all)
        # exit()
        contrastive_loss = loss_all

        # contrastive_loss = - torch.mean(torch.log(pos_scores_3 + 1e-8) + torch.log(1 - neg_scores_3 + 1e-8)) / 2
        # generative_loss_1 = torch.mean(torch.sum(torch.square(generative_diff_1), dim = 1))
        # generative_loss_2 = torch.mean(torch.sum(torch.square(generative_diff_2), dim = 1))
        generative_loss_1 = torch.mean(torch.square(generative_diff_1))
        generative_loss_2 = torch.mean(torch.square(generative_diff_2))
        
        generative_loss = (generative_loss_1 + generative_loss_2) / 2
        # print(raw_feat_3[::4, :][:5, :5])
        # print(generative_diff_1[:5, :5])
        # print(generative_loss_1)
        # print(generative_loss_2)
        # print(generative_loss)
        # exit()
        total_loss = self.alpha * contrastive_loss + self.beta * generative_loss

        contrastive_score_1 = neg_scores_1 - pos_scores_1
        contrastive_score_2 = neg_scores_2 - pos_scores_2
        contrastive_score = (contrastive_score_1 + contrastive_score_2) / 2
        contrastive_score = neg_scores_3 - pos_scores_3
        
        # print(pos_scores_3[:20])
        # print(neg_scores_3[:20])
        # exit()

        generative_score_1 = torch.sqrt(torch.sum(torch.square(generative_diff_1), dim = 1))
        generative_score_2 = torch.sqrt(torch.sum(torch.square(generative_diff_2), dim = 1))
        generative_score = (generative_score_1 + generative_score_2) / 2
        # print(generative_score_1[:10])
        # print(generative_score_2[:10])
        # print(generative_score[:10])
        # exit()

        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()
        # print(type(contrastive_score))
        # print(contrastive_score.shape)
        # print(contrastive_score[:10])
        # print(generative_score[:10])
        # print(type(contrastive_score.cpu().detach().numpy().reshape(-1, 1)))
        # print(contrastive_score.cpu().detach().numpy().reshape(-1, 1).shape)
        
        # exit()
        contrastive_score = scaler1.fit_transform(contrastive_score.cpu().detach().numpy().reshape(-1, 1))
        generative_score = scaler2.fit_transform(generative_score.cpu().detach().numpy().reshape(-1, 1))
        contrastive_score = torch.from_numpy(contrastive_score).reshape(anchor_embs.shape[0], 1)
        generative_score = torch.from_numpy(generative_score).reshape(anchor_embs.shape[0], 1)
        # print(type(contrastive_score))
        # print(contrastive_score.shape)
        # print(type(generative_score))
        # print(generative_score.shape)
        # print(contrastive_score[:20])
        # print(generative_score[:20])
        # exit()
        total_score = self.alpha * contrastive_score + self.beta * generative_score
        # print(total_score.dtype)
        # print(total_loss.dtype)
        # exit()
        # print(neg_scores_1[:3])
        # print(pos_scores_1[:3])
        # print(anchor_out_1[:3, :5])
        # print(anchor_out_2[:3, :5])
        # print(pos_pool_emb_1[:3, :5])
        # print(contrastive_loss)
        # print(generative_loss)
        # print(total_loss)
        # print(contrastive_score[:10])
        # print(generative_score[:10])
        # exit()
        cri = torch.any(torch.isnan(contrastive_loss)) + torch.any(torch.isinf(contrastive_loss))
        if cri > 0:
            print('in_feat : ', torch.any(torch.isnan(pos_in_feat_1)))
            print('pos_pool : ', torch.any(torch.isnan(pos_pool_emb_1)))
            print('anchor_out : ', torch.any(torch.isnan(anchor_out_1)))

        return total_loss, total_score, contrastive_loss, generative_loss#, contrastive_score, generative_score

        ori_feat_1 = []
        ori_feat_2 = []
        
        # print("before unbatchg", time.time())
        # Time_Process.global_time.process_Time("before unbatchg")
        # print(pos_batchg_1)
        anchor_feat_test = pos_batchg_1.ndata['feat'][0, :].clone()
        # print(anchor_feat_test)
        # exit()
        unbatchg = dgl.unbatch(pos_batchg_1)
        # anchor_feat_test = unbatchg[:].ndata['feat'][0, :].clone()
        # print(anchor_feat_test)
        # print()
        # print(unbatchg[0])
        for g in unbatchg:
            anchor_feat = g.ndata['feat'][0, :].clone()
            ori_feat_1.append(anchor_feat)
        # print(ori_feat_1)
        # exit()
        unbatchg = dgl.unbatch(pos_batchg_2)
        for g in unbatchg:
            anchor_feat = g.ndata['feat'][0, :].clone()
            ori_feat_2.append(anchor_feat)

        # print("after unbatchg", time.time())
        # Time_Process.global_time.process_Time("after batchg")


        # print("before net train", time.time())
        # Time_Process.global_time.process_Time("before net train")

        # print("pos_batchg_1.device", pos_batchg_1.device)
        # print("pos_batchg_2.device", pos_batchg_2.device)
        # print("pos_in_feat_1.device", pos_in_feat_1.device)
        # print("pos_in_feat_2.device", pos_in_feat_2.device)
        
        feat_1, pos_pool_emb_1, anchor_out_1 = self.enc(pos_batchg_1, pos_in_feat_1, self.attention)
        # print("positive subgraph encoder one", time.time())
        # Time_Process.global_time.process_Time("positive subgraph encoder one")

        feat_2, pos_pool_emb_2, anchor_out_2 = self.enc(pos_batchg_2, pos_in_feat_2, self.attention)
        # print("positive subgraph encoder two", time.time())
        # Time_Process.global_time.process_Time("positive subgraph encoder two")
        feat_3, neg_pool_emb, _ = self.enc(neg_batchg, neg_in_feat, self.attention)
        # print("negative subgraph encoder", time.time())
        # Time_Process.global_time.process_Time("negative subgraph encoder")
        # print(pos_pool_emb_1)
        # print(anchor_out_1)
        # print(pos_pool_emb_1.shape)
        # print(anchor_out_1.shape)
        # print("pos_pool_emb_1.device", pos_pool_emb_1.device)
        # print("anchor_out_1.device", anchor_out_1.device)
        
        pos_scores_1 = self.discriminator(pos_pool_emb_1, anchor_out_1)
        pos_scores_2 = self.discriminator(pos_pool_emb_2, anchor_out_2)
        neg_scores_1 = self.discriminator(neg_pool_emb, anchor_out_1)
        neg_scores_2 = self.discriminator(neg_pool_emb, anchor_out_2)
        # print("discriminator", time.time())
        # Time_Process.global_time.process_Time("discriminator")

        # print("pos_scores_1.device", pos_scores_1.device)
        # print("pos_scores_1", pos_scores_1)
        # print("pos_scores_2", pos_scores_2)
        # print("neg_scores_1", neg_scores_1)
        # print("neg_scores_2", neg_scores_2)
        # exit()
        pos_scores_1 = torch.sigmoid(pos_scores_1)
        pos_scores_2 = torch.sigmoid(pos_scores_2)
        neg_scores_1 = torch.sigmoid(neg_scores_1)
        neg_scores_2 = torch.sigmoid(neg_scores_2)
        # print("sigmoid", time.time())
        # Time_Process.global_time.process_Time("sigmoid")

        # pos_scores_1 = torch.sigmoid(pos_scores_1)
        # pos_scores_2 = torch.sigmoid(pos_scores_2)
        # neg_scores_1 = torch.sigmoid(neg_scores_1)
        # neg_scores_2 = torch.sigmoid(neg_scores_2)
        # print("")
        # print(pos_in_feat_1.shape)
        # print(pos_pool_emb_1.shape)
        # print(anchor_out_1.shape)
        # print(feat_1.shape)
        # exit()
        with pos_batchg_1.local_scope():
            # pos_batchg_1.ndata['feat'] = feat_1
            X_hat_1, subgraph_pool_emb_1, anchor_out_1 = self.dec(pos_batchg_1, feat_1)
        # print("positive subgraph decoder one", time.time())
        # Time_Process.global_time.process_Time("positive subgraph decoder one")
        with pos_batchg_2.local_scope():
            # pos_batchg_2.ndata['feat'] = feat_2
            X_hat_2, subgraph_pool_emb_2, anchor_out_2 = self.dec(pos_batchg_2, feat_2)
        # print("positive subgraph decoder two", time.time())
        # Time_Process.global_time.process_Time("positive subgraph decoder two")
        # print("anchor_out_1.device", anchor_out_1.device)
        # return pos_scores[:, 0], neg_scores[:, 0]
        
        # alpha = 1
        # beta = [0.2, 0.4, 0.6, 0.8, 1.0][2]
        # print("before net loss", time.time())
        # Time_Process.global_time.process_Time("before net loss")
        alpha = args.alpha
        beta = args.beta

        L_con = 0
        # pos_scores_1 = pos_scores_1.squeeze(1).cpu().detach().numpy()
        # pos_scores_2 = pos_scores_2.squeeze(1).cpu().detach().numpy()
        # neg_scores_1 = neg_scores_1.squeeze(1).cpu().detach().numpy()
        # neg_scores_2 = neg_scores_2.squeeze(1).cpu().detach().numpy()

        # print(pos_scores_1.shape)
        # print(neg_scores_1.shape)
        
        L_con = 0
        # L_con = L_con + numpy.mean(math.log(pos_scores_1) + math.log(1 - neg_scores_1)) / 2
        # L_con = L_con + numpy.mean(math.log(pos_scores_2) + math.log(1 - neg_scores_2)) / 2
        L_con = L_con + torch.mean(torch.log(pos_scores_1) + torch.log(1 - neg_scores_1)) / 2
        L_con = L_con + torch.mean(torch.log(pos_scores_2) + torch.log(1 - neg_scores_2)) / 2
        L_con = L_con / 2
        L_con = -L_con
        L_gen = 0
        # L_gen = L_gen + numpy.mean(numpy.square(X_hat_1 - pos_in_feat_1))
        # L_gen = L_gen + numpy.mean(numpy.square(X_hat_2 - pos_in_feat_2))

        # print(ori_feat_1[0])
        # print(ori_feat_1)
        # exit()

        # print(anchor_out_1.shape)
        # print()
        # print(ori_feat_1)
        
        ori_feat_1 = torch.stack(ori_feat_1, dim = 0)
        ori_feat_2 = torch.stack(ori_feat_2, dim = 0)
        
        # ori_feat_1 = torch.tensor(ori_feat_1)#.to('cuda:0')
        # ori_feat_2 = torch.tensor(ori_feat_2)#.to('cuda:0')
        # print(ori_feat_1.shape)
        L_gen = L_gen + torch.mean(torch.sum(torch.square(anchor_out_1 - ori_feat_1), dim = 1))
        L_gen = L_gen + torch.mean(torch.sum(torch.square(anchor_out_2 - ori_feat_2), dim = 1))
        L_gen = L_gen / 2
        # print(pos_scores_1)
        # print(neg_scores_1)
        # print(pos_scores_2)
        # print(neg_scores_2)
        # print(L_con)
        # print(L_gen)
        # exit()
        L = alpha * L_con + beta * L_gen
        # print(pos_scores_1.shape)
        # print((anchor_out_1 - ori_feat_1).shape)
        # exit()
        single_predict_scores = 0
        attributes_num = math.sqrt(len(anchor_out_1[0]))
        # print(attributes_num)
        # exit()
        contrastive_predict_scores = 0
        generative_predict_scores = 0

        contrastive_predict_scores = contrastive_predict_scores + (neg_scores_1 - pos_scores_1 + 1) / 2
        contrastive_predict_scores = contrastive_predict_scores + (neg_scores_1 - pos_scores_2 + 1) / 2
        generative_predict_scores = generative_predict_scores + torch.norm(anchor_out_1 - ori_feat_1, dim = 1, keepdim = True) / attributes_num / 2
        generative_predict_scores = generative_predict_scores + torch.norm(anchor_out_2 - ori_feat_2, dim = 1, keepdim = True) / attributes_num / 2
        # print(contrastive_predict_scores)
        # print(generative_predict_scores)

        print(contrastive_predict_scores.shape)
        print(generative_predict_scores.shape)

        single_predict_scores = alpha * contrastive_predict_scores + beta * generative_predict_scores
        print(single_predict_scores.shape)
        # exit()
        # single_predict_scores = single_predict_scores + torch.norm(anchor_out_1 - ori_feat_1) / attributes_num / 2
        # single_predict_scores = single_predict_scores + torch.norm(anchor_out_2 - ori_feat_2) / attributes_num / 2
        # single_predict_scores = single_predict_scores + (neg_scores_1 - pos_scores_1 + 1) / 2
        # single_predict_scores = single_predict_scores + (neg_scores_2 - pos_scores_2 + 1) / 2
        

        # print("net done", time.time())
        # Time_Process.global_time.process_Time("net done")
        return L, single_predict_scores

from .dataset import SL_GAD_DataSet
from .SL_GAD_utils import train_epoch, test_epoch
from dgl.dataloading import GraphDataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class SL_GAD():    
    def __init__(self, in_feats=1433, out_feats=64, global_adg=True, alpha = 1.0, beta = 0.6, args = None):
        """
        Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection
        Yu IEEE Transactions on Knowledge and Data Engineering 2021

        Parameters
        ----------
        in_feats : Torch.tensor, optional
            the feature dimensions of input data, default 1433
        out_feats : Torch.tensor, optional
            the feature dimensions of output data, default 64
        global_adg : bool, optional
            whether use the global information of node, here means the degree matrix, default True
        alpha : double, optional
            the coefficient of contrastive loss and score, default 1.0
        beta : double, optional
            the coefficient of generateive loss and score, default 0.6
        args : parser, optional
            extra custom made of model, default None

        Examples:
        ---------
        ```python
        >>> from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
        >>> from DGLD.CoLA import CoLA
        >>> if __name__ == '__main__':
        >>>     # sklearn-like API for most users.
        >>>     gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
        >>>     g = gnd_dataset[0]
        >>>     label = gnd_dataset.anomaly_label
        >>>     model = SL_GAD(in_feats=1433)
        >>>     model.fit(g, num_epoch=1, device='cpu')
        >>>     result = model.predict(g, auc_test_rounds=2)
        >>>     print(split_auc(label, result))
        ```
        """        
        self.args = args
        self.model = SL_GAD_Model(in_feats, out_feats, global_adg, args = args)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def fit(self, g, device='cpu', batch_size=300, lr=0.003, weight_decay=1e-5, num_workers=4, num_epoch=100, logdir='tmp', seed=42):
        """
        Parameters
        ----------
        g : DGL.Graph
            input graph with feature named "feat" in g.ndata
        device : str, optional
            device, default 'cpu'
        batch_size : int, optional
            batch size for training, default 300
        lr : float, optional
            learning rate for training, default 0.003
        weight_decay : float, optional
            weight decay for training, default 1e-5
        num_workers : int, optional
            num_workers using in `pytorch DataLoader`, default 4
        num_epoch : int, optional
            number of epoch for training, default 100
        logdir : str, optional
            tensorboard logdir, default 'tmp'
        seed : int, optional
            random seed, default 42
            
        Returns
        -------
        self : mpdel
            return the model.
        """        
        dataset = SL_GAD_DataSet(base_dataset_name = 'custom', g_data = g)
        print(dataset.dataset)
        # exit()
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
                epoch = epoch, loader = train_loader, net = self.model, device = device, criterion = self.criterion, optimizer = optimizer, args = self.args
            )
            writer.add_scalar("loss", float(loss_accum), epoch)
        return self

    def predict(self, g, device='cpu', batch_size=300, num_workers=4, auc_test_rounds=256, logdir='tmp'):
        """
        Parameters
        ----------
        g : DGL.Graph
            input graph with feature named "feat" in g.ndata.
        device : str, optional
            device, default 'cpu'
        batch_size : int, optional
            batch size for predicting, default 300
        num_workers : int, optional
            num_workers using in pytorch DataLoader, default 4
        auc_test_rounds : int, optional
            number of epoch for predciting, default 256
        logdir : str, optional
            tensorboard logdir, default 'tmp'
        Returns
        -------
        predict_score_arr : Torch.tensor
            the anomaly score of anchor nodes
        """

        dataset = SL_GAD_DataSet(base_dataset_name = 'custom', g_data = g)
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
                epoch = rnd, loader = test_loader, net = self.model, device = device, criterion = self.criterion, args = self.args, optimizer = None
            )
            predict_score_arr.append(list(predict_score))
        predict_score_arr = np.array(predict_score_arr).T
        return predict_score_arr.mean(1)





if __name__ == "__main__":
    # sample
    model = SL_GAD_Model(5)
    g1 = dgl.graph(([1, 2, 3], [2, 3, 1]))
    g1 = dgl.add_self_loop(g1)
    g1.ndata["feat"] = torch.rand((4, 5))
    g2 = dgl.graph(([3, 2, 4], [2, 3, 1]))
    g2 = dgl.add_self_loop(g2)
    g2.ndata["feat"] = torch.rand((5, 5))
    bg = dgl.batch([g1, g2])
    bg2 = dgl.batch([g2, g1])

    ans = model(bg, bg.ndata["feat"], bg2, bg2.ndata["feat"])
    print(ans)
    # g.ndata['feat'] = torch.rand((4, 5))
    # print(g.ndata['feat'])
    # subg = dgl.node_subgraph(g, [1,2])
    # print(subg.ndata['feat'])
    # subg.ndata['feat'] = torch.zeros((2, 5))
    # print(subg.ndata['feat'])
    # print(g.ndata['feat'])
