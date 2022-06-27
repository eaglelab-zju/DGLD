from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.ComGA import ComGA
from DGLD.ComGA import get_parse
from DGLD.common.evaluation import split_auc
from DGLD.common.utils import load_ACM
from DGLD.utils.utils import seed_everything
import dgl
import torch
import numpy as np

if __name__ == '__main__':
    # """
    # sklearn-like API for most users.
    # """
    # """
    # using GraphNodeAnomalyDectionDataset 
    # """
    # gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
    # g = gnd_dataset[0]
    # label = gnd_dataset.anomaly_label
    # model = ComGA(num_nodes=2708,num_feats=1433,n_enc_1=2000,n_enc_2=500,n_enc_3=128,dropout=0.0)
    # model.fit(g, num_epoch=1, device='cpu')
    # result = model.predict(g,device='cpu')
    # print(split_auc(label, result))

    # """
    # custom dataset
    # """
    # g=load_ACM()[0]
    # label = g.ndata['label']
    # model = ComGA(num_nodes=16484,num_feats=8337,n_enc_1=2000,n_enc_2=500,n_enc_3=128,dropout=0.0)
    # model.fit(g, num_epoch=1, device='2')
    # result = model.predict(g,device='2')
    # print(split_auc(label, result))
    """[command line mode]
    test command line mode
    """
    args = get_parse()
    seed_everything(args['seed'])
    gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = ComGA(**args["model"])
    model.fit(g, **args["fit"])
    result = model.predict(g, **args["predict"])
    split_auc(label, result)
    print(args)
