import sys
import os
current_file_name = __file__
CODE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
sys.path.append(CODE_DIR)

import torch
import dgl
import numpy as np

from typing import Literal

from dgld.utils.load_data import *
from dgld.utils.common_params import *
from dgld.utils.inject_anomalies import *

class NodeLevelAnomalyDataset(dgl.data.DGLDataset):
    """Generally, existing node-level graph anomaly detection datasets can be classified as follows:
    
    1. Downsampled node classification datasets.
    
    The widely used node classification datasets can be easily converted into sets suitable for anomaly detection through two steps. There are two ways:

    - Firstly, one particular class and its data records are chosen to represent normal objects. Then, the other data records are downsampled as anomalies at a specified downsampling rate.
    - Firstly, one particular class and its data records are chosen to represent abnormal objects. Then, its data records are downsampled as anomalies at a specified downsampling rate.

    By this, the generated node anomaly detection dataset is, in fact, a subset of the original dataset.  The most significant strength of this strategy is that no single data record has been modified.
    
    2. Real-world datasets with injected anomalies.
    
    These datasets are built based on the real-world networks. In particular, anomalies are generated either by perturbing the topological structure or the attributes of existing nodes/edges/sub-graphs, or by inserting non-existent graph objects.
    
    3. Datasets containing ground truth of natural anomalies. 
    
    These datasets contain anomalies that occur in the real world due to natural causes rather than being artificially injected. These datasets are useful for evaluating the performance of anomaly detection algorithms in real-world scenarios.
    
    In [A Comprehensive Survey on Graph Anomaly Detection with Deep Learning](https://arxiv.org/pdf/2106.07178.pdf), there is a more detailed introduction to the data set of graph anomaly detection.

    """
    def __init__(self, name, category: Literal['injected', 'natural', 'downsampled'], raw_dir=None, random_seed=42, verbose=True, transform=None, **kwargs):
        natural_list = ['weibo', 'Amazon', 'Enron', 'reddit', 'wiki', 'tfinance', 'elliptic', 'tsocial', 'Disney', 'reddit2', 'dgraphfin']
        injected_list = ['Cora', 'Citeseer', 'Pubmed', 'BlogCatalog', 'Flickr', 'ACM', 'ogbn-arxiv', 'Computers']
        downsampeled_list = ['Cora', 'Citeseer', 'Pubmed']
        assert (name in natural_list) or (name in injected_list) or (name in downsampeled_list), "Please enter valid data category including ['injected', 'natural', 'downsampled']."
        
        # set random seed
        np.random.seed(random_seed)
        
        if category == 'injected':
            graph = load_data(name, feat_norm=False, raw_dir=raw_dir)
            
            graph = inject_contextual_anomalies(graph=graph,k=K,p=P,q=Q_MAP[name],seed=42)
            graph = inject_structural_anomalies(graph=graph,p=P,q=Q_MAP[name],seed=42)
            graph = [graph]
        elif category == 'natural':
            graph = load_truth_data(data_path=raw_dir, dataset_name=name)
            graph = [graph]
        elif category == 'downsampled':
            if name in dgl_datasets:
                graph = getattr(dgl.data, name + 'GraphDataset')(raw_dir = raw_dir)[0]
            elif name in ogb_datasets:
                graph = load_ogbn_arxiv(raw_dir=raw_dir)
                
            graph = self.downsampled_all_class(graph, rate=kwargs.get('downsampled_rate'))
        
        # graph.ndata['feat'] = graph.ndata['feat'].float()
        self.graph = graph
        
        super().__init__(name=name, raw_dir=raw_dir, hash_key=(random_seed), verbose=verbose, transform=transform)
    
    def downsampled_all_class(self, graph, rate=0.1):
        original_label = graph.ndata['label']
        num_classes = torch.unique(original_label).shape[0]
        
        glst = []
        for class_id in list(np.unique(original_label)):
            ng = self.downsampled(graph.clone(), anomaly_class=int(class_id), rate=rate)
            glst.append(ng)
        
        return glst
        
    def downsampled(self, graph, anomaly_class=None, normal_class=None, rate=0.1):
        """There are two ways:

        - Firstly, one particular class and its data records are chosen to represent normal objects. Then, the other data records are downsampled as anomalies at a specified downsampling rate.
        - Firstly, one particular class and its data records are chosen to represent abnormal objects. Then, its data records are downsampled as anomalies at a specified downsampling rate.
        
        """
        original_label = graph.ndata['label']
        anomaly_label = torch.zeros_like(original_label)
        
        assert (anomaly_class is not None) or (normal_class is not None)
        
        if isinstance(anomaly_class, int):
            candidate_idx = np.where(original_label == anomaly_class)[0]
            anomaly_idx = np.random.choice(candidate_idx, size=int(rate*candidate_idx.shape[0]), replace=False)
            
        if isinstance(normal_class, int):
            candidate_idx = np.where(original_label != normal_class)[0]
            anomaly_idx = np.random.choice(candidate_idx, size=int(rate*candidate_idx.shape[0]), replace=False)
            
        anomaly_label[anomaly_idx] = 1
        graph.ndata['label'] = anomaly_label
        remove_idx = np.setdiff1d(candidate_idx, anomaly_idx)
        graph = dgl.remove_nodes(graph, remove_idx) 
        
        return graph
        
    
    def __getitem__(self, idx):
        r""" Get graph object

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features, node labels

            - ``ndata['feature']``: node features
            - ``ndata['label']``: node labels
        """
        if isinstance(self.graph, list):
            g = self.graph[idx]
        else:
            assert idx == 0, "This dataset has only one graph"
            g = self.graph
            
        if self._transform is None:
            return g
        else:
            return self._transform(g)
        
    def __len__(self):
        """number of data examples"""
        return len(self.graph)
    
if __name__ == '__main__':
    data_path = os.path.join(CODE_DIR, 'dgld/data/downloads')
    
    print('-'*20, 'downsampled', '-'*20)
    dataset = NodeLevelAnomalyDataset('Cora', category='downsampled', raw_dir=data_path, downsampled_rate=0.1)
    print(dataset)
    for graph in dataset:
        graph = dataset[0]
        # print(graph)
        label = graph.ndata['label']
        print(label.sum() / graph.num_nodes())
    
    print('-'*20, 'injected', '-'*20)
    dataset = NodeLevelAnomalyDataset('Cora', category='injected', raw_dir=data_path)
    print(dataset)
    graph = dataset[0]
    label = graph.ndata['label']
    print(label.sum())
    
    print('-'*20, 'natural', '-'*20)
    dataset = NodeLevelAnomalyDataset('Amazon', category='natural', raw_dir=data_path)
    print(dataset)
    graph = dataset[0]
    label = graph.ndata['label']
    print(label.sum())