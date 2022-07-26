import numpy as np
import torch
import dgl
from scipy.spatial.distance import euclidean


def inject_structural_anomalies(graph,p,q,seed=42):
    """
    Functions that inject structural anomaly
    """
    np.random.seed(seed)
    src, dst = graph.edges()
    labels = graph.ndata['label']

    number_nodes = graph.num_nodes()
    anomalies = set(torch.where(labels != 0)[0].numpy())

    new_src, new_dst = [], []
    # q cliques
    for i in range(q):
        q_list = []
        # selet p nodes
        for j in range(p):
            a = np.random.randint(number_nodes)
            while a in anomalies:
                a = np.random.randint(number_nodes)
            q_list.append(a)
            anomalies.add(a)
            labels[a] = 1
        # make full connected
        for n1 in range(p):
            for n2 in range(n1 + 1, p):
                new_src.extend([q_list[n1], q_list[n2]])
                new_dst.extend([q_list[n2], q_list[n1]])

    src, dst = list(src.numpy()), list(dst.numpy())
    src.extend(new_src)
    dst.extend(new_dst)
    # update edges
    graph.remove_edges(torch.arange(graph.num_edges()))
    graph.add_edges(src, dst)
    # print(graph.num_edges())
    # BUG
    r"""
    dgl.DGLGraph.to_simple is not supported inplace
    """
    # graph.to_simple()
    graph = dgl.to_simple(graph)
    # print(graph.num_edges())
    graph.ndata['label'] = labels
    structural_anomalies = torch.where(labels == 1)[0].numpy()
    print(
        "inject structural_anomalies numbers:", len(structural_anomalies)
    )
    anomalies = torch.where(labels != 0)[0].numpy()
    print("anomalies numbers:", len(anomalies))
    return graph

def inject_contextual_anomalies(graph,k,p,q,seed=42):
    """
    Functions that inject contextual anomaly
        
    """
    np.random.seed(seed)
    attribute_anomalies_number = p * q
    labels = graph.ndata['label']
    normal_nodes_idx = torch.where(labels == 0)[0].numpy()
    attribute_anomalies_idx = np.random.choice(
        normal_nodes_idx, size=attribute_anomalies_number, replace=False
    )
    all_attr = graph.ndata['feat']
    all_nodes_idx = list(range(graph.num_nodes()))
    for aa_i in attribute_anomalies_idx:
        # random sample k nodes
        random_k_idx = np.random.choice(all_nodes_idx, size=k, replace=False)
        # cal the euclidean distance and replace the node attribute with \
        biggest_distance = 0
        biggest_attr = 0
        for i in random_k_idx:
            dis = euclidean(all_attr[aa_i], all_attr[i])
            if dis > biggest_distance:
                biggest_distance, biggest_attr = dis, all_attr[i]
        # the node which has biggest one euclidean distance
        all_attr[aa_i] = biggest_attr

    graph.ndata['feat'] = all_attr
    labels[attribute_anomalies_idx] = 2
    graph.ndata['label'] = labels
    contextual_anomalies = torch.where(labels == 2)[0].numpy()
    print(
        "inject contextual_anomalies numbers:", len(contextual_anomalies)
    )
    anomalies = torch.where(labels != 0)[0].numpy()

    print("anomalies numbers:", len(anomalies))

    return graph