import os
import sys
current_file_name = __file__
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))) + '/src'
sys.path.append(current_dir)
print(current_dir)

from dgld.utils.dataset import GraphNodeAnomalyDectionDataset
from dgld.models.AnomalyDAE import AnomalyDAE
from dgld.models.AnomalyDAE import get_parse
from dgld.utils.evaluation import split_auc
from dgld.utils.common import load_ACM
from dgld.utils.common import seed_everything
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
    # model = AnomalyDAE(feat_size=1433,num_nodes=2708,embed_dim=256,out_dim=128,dropout=0.0)
    # model.fit(g, num_epoch=1, device='cpu')
    # result = model.predict(g,device='cpu')
    # print(split_auc(label, result))

    # """
    # custom dataset
    # """
    # g=load_ACM()[0]
    # label = g.ndata['label']
    # model = AnomalyDAE(feat_size=8337,num_nodes=16484,embed_dim=256,out_dim=128,dropout=0.0)
    # model.fit(g, num_epoch=1, device='0')
    # result = model.predict(g,device='0')
    # print(split_auc(label, result))
    """[command line mode]
    test command line mode
    """
    args = get_parse()
    seed_everything(args['seed'])
    gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = AnomalyDAE(**args["model"])
    model.fit(g, **args["fit"])
    result = model.predict(g, **args["predict"])
    split_auc(label, result)
    print(args)
