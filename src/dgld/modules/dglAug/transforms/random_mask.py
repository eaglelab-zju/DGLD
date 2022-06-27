"""random mask"""
from dgl import BaseTransform

try:
    import torch
    from torch.distributions import Bernoulli
except ImportError:
    pass


class RandomMask(BaseTransform):
    r"""Augment features by randomly masking node feautres with 0.

    Parameters
    ----------
    p : float, optional
        Probability of a node feautre to be masked.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dglAug import RandomMask

    >>> transform = RandomMask(p=0.5)
    >>> g = dgl.rand_graph(4,2)
    >>> g.ndata['feat'] = torch.rand((4,5))
    >>> print(g.ndata['feat'])
    tensor([[0.6242, 0.5736, 0.0784, 0.7627, 0.0377],
            [0.1672, 0.7696, 0.5750, 0.6666, 0.4387],
            [0.4001, 0.4118, 0.6463, 0.9568, 0.3902],
            [0.9920, 0.9099, 0.5543, 0.6682, 0.2897]])
    >>> g = transform(g)
    >>> print(g.ndata['feat'])
    tensor([[0.6242, 0.0000, 0.0000, 0.0000, 0.0377],
            [0.1672, 0.0000, 0.0000, 0.0000, 0.4387],
            [0.4001, 0.0000, 0.0000, 0.0000, 0.3902],
            [0.9920, 0.0000, 0.0000, 0.0000, 0.2897]])
    """
    def __init__(self, p=0.5):
        self.p = p
        self.dist = Bernoulli(p)

    def __call__(self, g):
        if self.p == 0:
            return g
        feat = g.ndata['feat']
        samples = self.dist.sample(torch.Size([feat.shape[1]]))
        drop_mask = samples.bool().to(g.device)
        feat[:, drop_mask] = 0
        g.ndata['feat'] = feat
        return g


# if __name__=='__main__':
#     transform = RandomMask(p=0.5)
#     import dgl
#     g = dgl.rand_graph(4,2)
#     g.ndata['feat'] = torch.rand((4,5))
#     print(g.ndata['feat'])
#     g=transform(g)
#     print(g.ndata['feat'])
