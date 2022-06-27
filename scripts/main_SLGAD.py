# Author: Peng Zhang <zzhangpeng@zju.edu.cn>
# License: BSD 2 clause
from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.SL_GAD import SL_GAD
from DGLD.SL_GAD import get_parse
from DGLD.common.evaluation import split_auc

import dgl
import torch
import numpy as np

if __name__ == '__main__':
    """
    sklearn-like API for most users.
    """
    """
    using GraphNodeAnomalyDectionDataset 
    """
    args = get_parse()
    gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = SL_GAD(in_feats=1433)
    model.fit(g, num_epoch=1, device='cpu')
    result = model.predict(g, auc_test_rounds=2)
    print(split_auc(label, result))

    """
    custom dataset
    """
    args = get_parse()
    g = dgl.graph((torch.tensor([0, 1, 2, 4, 6, 7]), torch.tensor([3, 4, 5, 2, 5, 2])))
    g.ndata['feat'] = torch.rand((8, 4))
    label = np.array([1, 2, 0, 0, 0, 0, 0, 0])
    model = SL_GAD(in_feats=4)
    model.fit(g, num_epoch=1, device='cpu')
    result = model.predict(g, auc_test_rounds=2)
    print(split_auc(label, result))
    
    """[command line mode]
    test command line mode
    """
    args = get_parse()
    print(args)
    
    gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = SL_GAD(**args["model"])
    model.fit(g, **args["fit"])
    result = model.predict(g, **args["predict"])
    print(split_auc(label, result))

