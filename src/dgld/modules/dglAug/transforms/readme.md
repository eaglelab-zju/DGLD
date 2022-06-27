# User-defined DGL graph transforms

## Define a graph data augmentation operator

- First, you need to inherit a base class `BaseTransform` from dgl.
- Then,you should implement two functions`__init__` and `__call__`.

```python
class MyTansform(BaseTransform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, g):
        if self.p == 0:
            return g
        # Implement your code
        return g

```

- Finally,you should import your transform in `transforms/__init__.py` like this `from .random_mask import RandomMask`
