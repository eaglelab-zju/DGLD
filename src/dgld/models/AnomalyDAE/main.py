
from ast import arg
import imp
import sys
import numpy as np

from sklearn.metrics import precision_score, roc_auc_score
sys.path.append('../../')

import scipy.sparse as sp
import scipy.io as sio
import torch
torch.set_printoptions(precision=8)
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import dgl
from anomalydae_utils import get_parse, train_step, test_step,normalize_adj
from models import AnomalyDAE
from common.dataset import GraphNodeAnomalyDectionDataset

if __name__ == '__main__':
    args = get_parse()
    print(args)
    # # load dataset
    dataset = GraphNodeAnomalyDectionDataset(args.dataset,cola_preprocess_features=False)
    graph = dataset[0]
    features = graph.ndata['feat']
    print(features)
    adj = graph.adj(scipy_fmt='csr')
    
    # # 构造原github数据为graph，进行测试
    # data_mat = sio.loadmat('/home/data/zp/ygm/AnomalyDAE/data/BlogCatalog/BlogCatalog.mat')
    # adj = data_mat['Network']
    # feat = data_mat['Attributes']
    # truth = data_mat['Label']
    # truth = truth.flatten()
    # print(np.sum(adj))
    # graph = dgl.from_scipy(adj)
    # graph=dgl.add_self_loop(graph)
    # print('loop graph:\n',graph)
    # features = torch.FloatTensor(feat.toarray())
    # print('features[1]',features[1])
    # adj = adj + sp.eye(adj.shape[0])
    

    # data preprocess
    adj_label = torch.FloatTensor(adj.toarray())

    print(graph)
    print('adj_label shape:', adj_label.shape)
    print('features shape:', features.shape)
    
    feat_dim=features.shape[1]
    num_nodes=features.shape[0]

    model = AnomalyDAE(in_feat_dim=feat_dim,in_num_dim=num_nodes,embed_dim=args.embed_dim,
                        out_dim=args.out_dim,dropout=args.dropout)

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda:" + str(args.device))
        print('Using gpu!!!')
    else:
        device = torch.device("cpu")
        print('Using cpu!!!')

    model = model.to(device)
    graph = graph.to(device)
    features = features.to(device)
    adj_label = adj_label.to(device)

    writer = SummaryWriter(log_dir=args.logdir)
    for epoch in range(args.num_epoch):
        loss, struct_loss, feat_loss,_ = train_step(
            args, model, optimizer, graph, features,adj_label,device)
        predict_score = test_step(args, model, graph, features, adj_label,device)
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item(
        )), "train/struct_loss=", "{:.5f}".format(struct_loss.item()), "train/feat_loss=", "{:.5f}".format(feat_loss.item()))
        writer.add_scalars(
            "loss",
            {"loss": loss, "struct_loss": struct_loss, "feat_loss": feat_loss},
            epoch,
        )
        # print('auc:',roc_auc_score(truth,predict_score))
        final_score, a_score, s_score = dataset.evalution(predict_score)
        writer.add_scalars(
            "auc",
            {"final": final_score, "structural": s_score, "attribute": a_score},
            epoch,
        )

        writer.flush()
