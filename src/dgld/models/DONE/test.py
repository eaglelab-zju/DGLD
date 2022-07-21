import torch
import dgl
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from models import DONE
from sklearn.metrics import roc_auc_score
import csv
from tqdm import tqdm

def read_csv_file_as_numpy(filepath:str):
    """Read csv file into numpy format

    Parameters
    ----------
    filepath : str
        file path
    """
    with open(filepath, 'r') as fp:
        rd = csv.reader(fp)
        ret = []
        for row in tqdm(rd):
            ret.append([float(r) for r in row])
    return np.array(ret)


def load_paper_dataset(dataset):
    """Loading the paper dataset

    Parameters
    ----------
    dataset : str
        The dataset used in the experiments of the paper
    """
    if dataset == 'cora':
        adj = read_csv_file_as_numpy('test_data/cora/A_Final_permuted.csv')
        feat = read_csv_file_as_numpy('test_data/cora/C_Final_permuted.csv')
        label = read_csv_file_as_numpy('test_data/cora/labels_Final_permuted.csv')
        indices = read_csv_file_as_numpy('test_data/cora/permutation.csv')
    else:
        adj = read_csv_file_as_numpy('test_data/{}/struct.csv'.format(dataset))
        feat = read_csv_file_as_numpy('test_data/{}/content.csv'.format(dataset))
        label = read_csv_file_as_numpy('test_data/{}/label.csv'.format(dataset))
        indices = read_csv_file_as_numpy('test_data/{}/permutation.csv'.format(dataset))
    print(adj.shape, feat.shape, label.shape)
    print("#Nodes: {}".format(adj.shape[0]))
    print("#Edges: {}".format(adj.nonzero()[0].shape[0]))
    print("#Labels: ", np.unique(label).shape[0])
    print("#Attributes: {}".format(feat.shape[1]))
    
    graph = dgl.graph(adj.nonzero())
    graph.ndata['feat'] = torch.FloatTensor(feat)
    graph.ndata['label'] = torch.IntTensor(label)
    
    return graph, indices

def recall_at_k(truth, score, k):
    ranking = np.argsort(-score)
    top_k = ranking[:k]
    top_k_label = truth[top_k]
    return top_k_label.sum() / truth.sum()

# 使用论文数据集和实验设置进行测试
dataset = 'cora'
g, indices = load_paper_dataset(dataset)
# newg = random_walk_with_restart(g)
feat = g.ndata['feat']
label = g.ndata.pop('label').squeeze()
label_aug = np.zeros_like(label)
indices = indices.squeeze()
# print(indices)
unseeded_num_dict = {
    'cora': 2708,
    'citeseer': 3312,
    'WebKB': 877,
}
label_aug[indices > unseeded_num_dict[dataset]] = 1
num_nodes = g.number_of_nodes()

model = DONE(feat.shape[1], num_nodes)
model.fit(g, batch_size=0, num_epoch=20, lr=0.03, y_true=label_aug)
score = model.predict(g, batch_size=0)

auc = roc_auc_score(label_aug, score)

print(f"auc: {auc:.5f}")
for p in [5, 10, 15, 20, 25]:
    k = int(num_nodes * p / 100.0)
    recall = recall_at_k(label_aug, score, k)
    print(f"Recall@{p}%: {recall:.5f}")