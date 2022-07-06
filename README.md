
<h1 align="center">
    <p>A Deep Graph Anomaly Detection Library based on DGL</p>
</h1>

DGLD is an open-source library for Deep Graph Anomaly Detection based on pytorch and DGL. It provides unified interface of popular graph anomaly detection methods, including the data loader, data augmentation, model training and evaluation. Also, the widely used modules are well organized so that developers and researchers can quickly implement their own designed models.


## Quick Start

Here, we introduce how to simply run DGLD, following 4 steps.

### Dataloader

We support multiple data import methods, including [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [DGL](https://www.dgl.ai/) and custom data. DGLD combines the process of data load and anomaly injection. Except for some basic datasets(including "Cora", "Citeseer", "Pubmed", "BlogCatalog", "Flickr", "ogbn-arxiv" and "ACM"), DGLD also accept custom data.

### Anomaly Injection

In anomaly detection, we inject the abnormal node in two methods, structural and contextual, by two parameters - p and k. gnd_dataset is an instance of GraphNodeAnomalyDectionDataset. g is an instance of DGL.Graph. label is an instnace of torch.Tensor, presenting the anomaly class. Following is an example showing that a few lines of codes are sufficient to load and inject.

```python
gnd_dataset = GraphNodeAnomalyDectionDataset("Cora", p = 15, k = 50)
g = gnd_dataset[0]
label = gnd_dataset.anomaly_label
```

### Model

DGLD supports some basic methods. It's easy to construct and train model.

```python
model = CoLA(in_feats = g.ndata['feat'].shape[1])
```

### Train and Evaluation

Function fit need parameters to specify number of epoch and device. For gpu, device should be a int, while a string 'cpu' for cpu.

```python
model.fit(g, num_epoch = 5, device = 0)
result = model.predict(g, auc_test_rounds = 2)
print(split_auc(label, result))
```

## Install
```shell
conda create -n dgld python=3.8.0
conda install cudatoolkit==11.3.1
pip install dgl-cu113==0.8.1 dglgo==0.0.1 -f https://data.dgl.ai/wheels/repo.html
pip install torch==1.11.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

```
## Native Datasets
The DGLD provides native graph anomaly detection datasets that widely used by existing methods.

|   Dataset   | nodes  |  edges  | attributes | anomalies |
| :---------: | :----: | :-----: | :--------: | :-------: |
| BlogCatalog |  5196  | 171743  |    8189    |    300    |
|   Flickr    |  7575  | 239738  |   12047    |    450    |
|     ACM     | 16484  |  71980  |    8337    |    600    |
|    Cora     |  2708  |  5429   |    1433    |    150    |
|  Citeseer   |  3327  |  4732   |    3703    |    150    |
|   Pubmed    | 19717  |  44338  |    500     |    600    |
| ogbn-arxiv  | 169343 | 1166243 |    128     |   6000    |



## Implemented Results



|Method   | BlogCatalog | Flickr  |  cora   | citeseer | pubmed  |   ACM   | ogbn-arxiv |
| :--------: | :---------: | :-----: | :-----: | :------: | :-----: | :-----: | :--------: |
| [CoLA]((https://arxiv.org/abs/2103.00113))    |   0.7854/   | 0.7513/ | 0.8779/ | 0.8968/  | 0.9512/ | 0.8237/ |  0.8073/   |
|  [SL-GAD](https://arxiv.org/pdf/2108.09896.pdf?ref=https://githubhelp.com)   |   0.8184/   | 0.7966/ | 0.9130/ | 0.9136/  | 0.9672/ | 0.8538/ |     /      |
|  [ANEMONE](https://dl.acm.org/doi/abs/10.1145/3459637.3482057)   |      /      |    /    | 0.9057/ | 0.9189/  | 0.9548/ |    /    |     /      |
| [DOMINANT](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67)  |   0.7813/0.5701   | 0.7490/0.5475 |    /0.9554    |    /0.8455     |    /oom    | 0.7494/oom |     /oom      |
|   [ComGA](https://dl.acm.org/doi/abs/10.1145/3488560.3498389)    |   0.814/    | 0.799/  | 0.884/  | 0.9167/  | 0.922/  | 0.8496/ |     /      |
| [AnomalyDAE](https://arxiv.org/pdf/2002.03665.pdf) |      0.9781/      |    0.9722/    |    /    |    /     |    /    |    0.9005/    |     /      |
|   [ALARM](https://ieeexplore.ieee.org/abstract/document/9162509)    |      /      |    /    |    /    |    /     |    /    |    /    |     /      |
|  [AAGNN](https://www4.comp.polyu.edu.hk/~xiaohuang/docs/Shuang_CIKM21.pdf)   |   0.8184/   | 0.8299/ |    /    |    /     | 0.8564/ |    /    |     /      |


## Citation