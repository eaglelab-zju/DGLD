"""Feature Dropout"""
import torch.nn.functional as F
from dgl import BaseTransform


class FeatureDropout(BaseTransform):
    r"""Augment features by randomly masking node feautres with 0.

    Parameters
    ----------
    p : float, optional
        Probability of a node feautre to be masked.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dglAug import FeatureDropout

    >>> transform = FeatureDropout(p=0.2)
    >>> g = dgl.rand_graph(4,2)
    >>> g.ndata['feat'] = torch.rand((4,5))
    >>> print(g.ndata['feat'])
    tensor([[0.7706, 0.3505, 0.1246, 0.5076, 0.3071],
        [0.5388, 0.6082, 0.5088, 0.8058, 0.4955],
        [0.7638, 0.3115, 0.4265, 0.5507, 0.4404],
        [0.3127, 0.0056, 0.1876, 0.9971, 0.6389]])
    >>> g = transform(g)
    >>> print(g.ndata['feat'])
    tensor([[0.0000, 0.0000, 0.1558, 0.6345, 0.3839],
        [0.0000, 0.7603, 0.6360, 0.0000, 0.6194],
        [0.0000, 0.3893, 0.5331, 0.0000, 0.5505],
        [0.3909, 0.0070, 0.2345, 1.2464, 0.0000]])
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, g):
        if self.p == 0:
            return g

        for ntype in g.ntypes:
            g.apply_nodes(
                lambda node: {'feat': F.dropout(node.data['feat'], self.p)},
                ntype=ntype,
            )
        return g


# if __name__ == '__main__':
#     transform = FeatureDropout(p=0.2)
#     import dgl
#     g = dgl.rand_graph(4, 2)
#     g.ndata['feat'] = torch.rand((4, 5))
#     print(g.ndata['feat'])
#     g = transform(g)
#     print(g.ndata['feat'])
