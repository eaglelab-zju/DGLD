"""Randomly shuffle"""
from dgl import backend as F
from dgl import BaseTransform


class NodeShuffle(BaseTransform):
    r"""Randomly shuffle the nodes.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import NodeShuffle

    >>> transform = NodeShuffle()
    >>> g = dgl.graph(([0, 1], [1, 2]))
    >>> g.ndata['h1'] = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> g.ndata['h2'] = torch.tensor([[7., 8.], [9., 10.], [11., 12.]])
    >>> g = transform(g)
    >>> print(g.ndata['h1'])
    tensor([[5., 6.],
            [3., 4.],
            [1., 2.]])
    >>> print(g.ndata['h2'])
    tensor([[11., 12.],
            [ 9., 10.],
            [ 7.,  8.]])
    """
    def __init__(self, is_use=True):
        self.is_use = is_use

    def __call__(self, g):
        if not self.is_use:
            return g
        for ntype in g.ntypes:
            nids = F.astype(g.nodes(ntype), F.int64)
            perm = F.rand_shuffle(nids)
            for key, feat in g.nodes[ntype].data.items():
                g.nodes[ntype].data[key] = feat[perm]
        return g
