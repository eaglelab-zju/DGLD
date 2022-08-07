""" Structural Clustering Algorithm for Networks
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import math
import numpy as np

import dgl
from dgld.utils.early_stopping import EarlyStopping


class SCAN(nn.Module):
    """
    SCAN (Structural Clustering Algorithm for Networks).

    Parameters
    ----------
    eps : float, optional
        Neighborhood threshold. Default: ``.5``.
    mu : int, optional
        Minimal size of clusters. Default: ``2``.

    Examples
    --------
    >>> from dgld.models.SCAN import SCAN
    >>> model = SCAN()
    >>> model.fit(g)
    >>> result = model.predict(g)
    """

    def __init__(self,
                 eps=.5,
                 mu=2):
        super(SCAN, self).__init__()

        # model param
        self.predict_score = None
        self.eps = eps
        self.mu = mu
        self.neighs = {}

    def fit(self, g):
        """Fitting model

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset.

        """
        decision_scores = np.zeros(g.num_nodes())

        # get nodes' neighbors
        adj = g.adj()._indices()

        for i in range(len(adj[0])):
            if not adj[0, i].item() in self.neighs:
                nei = []
                nei.append(adj[1, i].item())
                self.neighs[adj[0, i].item()] = nei
            else:
                self.neighs[adj[0, i].item()].append(adj[1, i].item())

        c = 0
        clusters = {}
        nomembers = []
        ind = 0
        for n in g.nodes():
            # print(ind, '/', g.num_nodes())
            ind += 1
            if self.hasLabel(clusters, n):
                continue
            else:
                N = self.eps_neighborhood(n.item())
                if len(N) > self.mu:
                    c = c + 1
                    Q = self.eps_neighborhood(n.item())
                    clusters[c] = []
                    # append core vertex itself
                    clusters[c].append(n)
                    while len(Q) != 0:
                        w = Q.pop(0)
                        R = self.eps_neighborhood(w)
                        # include current vertex itself
                        R.append(w)
                        for s in R:
                            if not (self.hasLabel(clusters, s)) or \
                                    s in nomembers:
                                clusters[c].append(s)
                            if not (self.hasLabel(clusters, s)):
                                Q.append(s)
                else:
                    nomembers.append(n)

        for k, v in clusters.items():
            decision_scores[v] = 1

        self.predict_score = decision_scores

    def similarity(self, v, u):
        """compute the similarity of two nodes' neighbors

        Parameters
        ----------
        v : int
            first node id.
        u : int
            second node id.

        Returns
        -------
        sim : float
            similarity of two nodes' neighbors.

        """
        v_set = set(self.neighs[v])
        u_set = set(self.neighs[u])
        inter = v_set.intersection(u_set)
        if inter == 0:
            return 0
        # need to account for vertex itself, add 2(1 for each vertex)
        sim = (len(inter) + 2) / (
            math.sqrt((len(v_set) + 1) * (len(u_set) + 1)))
        return sim

    def eps_neighborhood(self, v):
        """found eps-neighbors list

        Parameters
        ----------
        v : int
            node id.

        Returns
        -------
        eps_neighbors : list
            list of node's eps-neighbor

        """
        eps_neighbors = []
        v_list = self.neighs[v]
        for u in v_list:
            if (self.similarity(v, u)) > self.eps:
                eps_neighbors.append(u)
        return eps_neighbors

    def hasLabel(self, cliques, vertex):
        """judge whether the node is labeled

        Parameters
        ----------
        cliques : dict
            cluster dict.
        vertex : torch.tensor
            node id.

        Returns
        -------
        bool :
            whether the node is labeled.
        """
        for k, v in cliques.items():
            if vertex in v:
                return True
        return False

    def predict(self, g):
        """predict and return anomaly score of each node

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset.

        Returns
        -------
        score : numpy.ndarray
            anomaly score of each node.
        """
        print('*' * 20, 'predict', '*' * 20)

        score = self.predict_score

        return score
