import numpy
from scipy.io import loadmat
import scipy.sparse as sp

import sys
sys.path.append('./src')
from dgld.models.Radar import Radar
import torch
import dgl
from sklearn.metrics import roc_auc_score


dataset = 'Disney'
mat = loadmat(f'/Users/fangzeyu/Documents/图计算项目组/DGLD-1/src/dgld/models/Radar/dataset/{dataset}.mat')

label = mat['gnd']
A = mat['A']
X = mat['X']


print(mat.keys())
print(label.shape, A.shape, X.shape)
print(type(label), type(A), type(X))

if isinstance(A, numpy.ndarray):
    A = sp.csc_matrix(A)

graph = dgl.from_scipy(A)
graph.ndata['feat'] = torch.tensor(X).float()
print(graph)

model = Radar()

model.fit(graph, alpha=0.5, beta=0.2, gamma=0.2, num_epoch=100)
score = model.predict(graph)
auc = roc_auc_score(label, score)
print(auc)

# best = 0
# for alpha in [1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 1e2, 1e3]:
#     for beta in [1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 1e2, 1e3]:
#         for gamma in [1e-3, 1e-2, 1e-1, 0.2, 0.5, 1, 1e2, 1e3]:
#             print(alpha, beta, gamma)
#             model.fit(graph, alpha=alpha, beta=beta, gamma=gamma, num_epoch=100, verbose=False)
#             score = model.predict(graph)
#             auc = roc_auc_score(label, score)
#             print(auc)
#             best = max(auc, best)
            
# print(best)