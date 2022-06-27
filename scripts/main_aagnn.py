# Author: Peng Zhang <zzhangpeng@zju.edu.cn>
# License: BSD 2 clause
from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD.common.evaluation import split_auc
from DGLD.AAGNN import AAGNN
from DGLD.AAGNN import AAGNN_batch

from DGLD.AAGNN import get_parse
import dgl
import torch
import numpy as np

if __name__ == '__main__':
    """[command line mode]
    test command line mode
    """
    args = get_parse()
    print(args)
    gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    g = gnd_dataset[0]
    label = gnd_dataset.anomaly_label
    model = AAGNN_batch(**args["model"])
    model.fit(g, **args["fit"])
    result = model.predict(g, **args["predict"])
    split_auc(label, result)





