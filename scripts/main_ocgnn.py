import os
import sys
current_file_name = __file__
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))) + '/src'
sys.path.append(current_dir)
print(current_dir)

from dgld.utils.dataset import GraphNodeAnomalyDectionDataset
from dgld.models.OCGNN import OCGNN
from dgld.models.OCGNN import get_parse
from dgld.utils.evaluation import split_auc

import dgl
import torch
import numpy as np
import torch.nn.functional as F


if __name__ == '__main__':
    """
    sklearn-like API for most users.
    """
    """
    using GraphNodeAnomalyDectionDataset 
    """

    args = get_parse()
    # as experiment in paper, use normal data to train
    gtrain_dataset = GraphNodeAnomalyDectionDataset(args['dataset'], p=0, k=0)
    gtrain = gtrain_dataset[0]
    print('-------------------------')
    model = OCGNN(args['model']['feat_size'], args['model']['module'], args['model']['hidden_dim'],
                  args['model']['n_layers'], args['model']['dropout'], args['model']['nu'], act=F.relu)
    model.fit(gtrain, lr=5e-3, batch_size=0, num_epoch=300, warmup_epoch=1, weight_decay=0., device='0')
    print('-------------------------')
    # use data mixed with abnormal data to test
    gtest_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    g = gtest_dataset[0]
    label = gtest_dataset.anomaly_label
    result = model.predict(g, device='cpu')
    print(split_auc(label, result))

    """
    custom dataset
    """
    # g = dgl.graph((torch.tensor([0, 1, 2, 4, 6, 7]), torch.tensor([3, 4, 5, 2, 5, 2])))
    # g.ndata['feat'] = torch.rand((8, 4))
    # label = np.array([1, 2, 0, 0, 0, 0, 0, 0])
    # model = OCGNN(input_dim=4, module='GCN')
    # model.fit(g, num_epoch=1, device='cpu')
    # result = model.predict(g, device='cpu')
    # print(split_auc(label, result))

    """[command line mode]
    test command line mode
    """
    # args = get_parse()
    # print(args)
    #
    # gnd_dataset = GraphNodeAnomalyDectionDataset(args['dataset'])
    # g = gnd_dataset[0]
    # label = gnd_dataset.anomaly_label
    # model = OCGNN(**args["model"])
    # model.fit(g, **args["fit"])
    # result = model.predict(g, **args["predict"])
    # print(split_auc(label, result))