"""Graph Augmentation
Adapted from https://github.com/PyGCL/PyGCL/blob/main/GCL/augmentors/augmentor.py
"""
from copy import deepcopy
from typing import List

import dgl
import torch
from dgl import BaseTransform


# pylint:disable=no-else-return
class ComposeAug(BaseTransform):
    """Execute graph augments in sequence.

    Parameters
    ----------
    augs : List[BaseTransform]
        graphs augments using DGL tansform
    cross : bool, optional
        if use cross graph augments, by default True
    """
    def __init__(self, augs: List[BaseTransform], cross: bool = True) -> None:
        super().__init__()
        self.augs = augs
        self.cross = cross

    def __call__(self, g: dgl.DGLGraph):
        """Execute augments on graph

        Parameters
        ----------
        g : dgl.DGLGraph
            raw graph

        Returns
        -------
        if cross == True:
            return cross augmented graph
        else:
            return multiple augmented graphs
        """
        if self.cross:
            for aug in self.augs:
                g = aug(g)
            return g
        else:
            graphs = []
            tmpg = deepcopy(g)
            for aug in self.augs:
                newg = aug(tmpg)
                tmpg = deepcopy(g)
                graphs.append(newg)
            return graphs


class RandomChoiceAug(BaseTransform):
    """Execute graph augments in random.

    Parameters
    ----------
    augs : _type_
        _description_
    n_choices : _type_
        _description_
    cross : bool, optional
        _description_, by default True
    """
    def __init__(self,
                 augs: List[BaseTransform],
                 n_choices: int,
                 cross: bool = True) -> None:
        super().__init__()
        assert n_choices <= len(augs), 'n_choices should <= augs length'
        self.augs = augs
        self.n_choices = n_choices
        self.cross = cross

    def __call__(self, g):
        """Execute augments on graph

        Parameters
        ----------
        g : dgl.DGLGraph
            raw graph

        Returns
        -------
        if cross == True:
            return cross augmented graph
        else:
            return multiple augmented graphs
        """
        n_augs = len(self.augs)
        perm = torch.randperm(n_augs)
        idx = perm[:self.n_choices]

        if self.cross:
            for i in idx:
                aug = self.augs[i]
                g = aug(g)
            return g
        else:
            graphs = []
            tmpg = deepcopy(g)
            for i in idx:
                aug = self.augs[i]
                newg = aug(tmpg)
                tmpg = deepcopy(g)
                graphs.append(newg)
            return graphs
