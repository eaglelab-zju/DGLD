from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.DOMINANT import Dominant
from DGLD.DOMINANT import get_parse
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
    # model = Dominant(feat_size=1433, hidden_size=64, dropout=0.3)
    # model.fit(g, num_epoch=1, device='cpu')
    # result = model.predict(g)
    # print(split_auc(label, result))

    # """
    # custom dataset
    # """
    # gnd_dataset = GraphNodeAnomalyDectionDataset("ACM")
    # g = gnd_dataset[0]
    # label = gnd_dataset.anomaly_label
    # model = Dominant(feat_size=8337, hidden_size=64, dropout=0.3)
    # model.fit(g, num_epoch=1, device='4')
    # result = model.predict(g,device='4')
    # print(split_auc(label, result))
    """[command line mode]
    test command line mode
    """
    args = get_parse()
    seed_everything(args['seed'])
    gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = Dominant(**args["model"])
    model.fit(g, **args["fit"])
    result = model.predict(g, **args["predict"])
    split_auc(label, result)
    print(args)

