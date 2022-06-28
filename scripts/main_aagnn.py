import os
import sys
current_file_name = __file__
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))) + '/src'
sys.path.append(current_dir)
print(current_dir)

import dgld

from dgld.utils.dataset import GraphNodeAnomalyDectionDataset
from dgld.utils.evaluation import split_auc
from dgld.models.AAGNN import AAGNN_batch

from dgld.models.AAGNN import get_parse
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





