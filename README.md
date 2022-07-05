
<h1 align="center">
    <p>A Deep Graph Anomaly Detection Library based on DGL</p>
</h1>

DGLD is an open-source library for Deep Graph Anomaly Detection based on pytorch and DGL. It provides unified interface of popular graph anomaly detection methods, including the data loader, data augmentation, model training and evaluation. Also, the widely used modules are well organized so that developers and researchers can quickly implement their own designed models.

## Overview of Library
@ZS

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



## Implemented Methods

@GZN 这部分最终确认

|finished| Paper                                                                                                                                |  Method  |  From   |                                 Code                                 |
|:---| :----------------------------------------------------------------------------------------------------------------------------------- | :------: | :-----: | :------------------------------------------------------------------: |
|- [x] | [Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113)                |   CoLA   | TNNLS21 |         [Pytorch+DGL0.3](https://github.com/GRAND-Lab/CoLA)          |
|- [ ]| [Deep Anomaly Detection on Attributed Networks](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67)                           | Dominant |  SDM19  | [Pytorch](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch) |
|- [x]| [Subtractive Aggregation for Attributed Network Anomaly Detection](https://www4.comp.polyu.edu.hk/~xiaohuang/docs/Shuang_CIKM21.pdf) |  AAGNN   |  CIKM21   |                                                                      |
|- [ ]| [ComGA: Community-Aware Attributed Graph Anomaly Detection](https://dl.acm.org/doi/abs/10.1145/3488560.3498389) |  ComGA    |  WSDM22   |   [Tensorflow 1.0 ](https://github.com/DASE4/ComGA)   |
|- [ ]| [A Deep Multi-View Framework for Anomaly Detection on Attributed Networks](https://ieeexplore.ieee.org/abstract/document/9162509) |  ALARM     |  TKDE20   |     |
|- [ ]| [ANOMALYDAE: DUAL AUTOENCODER FOR ANOMALY DETECTION ON ATTRIBUTED NETWORKS](https://arxiv.org/pdf/2002.03665.pdf) |  AnomalyDAE     |  ICASSP20   |   [Tensorflow 1.10 ](https://github.com/haoyfan/AnomalyDAE)    |
|- [ ]| [Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection](https://arxiv.org/pdf/2108.09896.pdf?ref=https://githubhelp.com) |  SL-GAD     |  TKDE21   |     |
|- [ ]| [ANEMONE: Graph Anomaly Detection with Multi-Scale Contrastive Learning](https://dl.acm.org/doi/abs/10.1145/3459637.3482057) |  ANEMONE      |  CIKM21  |   [DGL0.4.1 ](https://github.com/GRAND-Lab/ANEMONE)    |





## Reproduced results 
@GZN 这个表格最终确认
Reported/Reproduced

|                 Reproducer                  |   Method   | BlogCatalog | Flickr  |  cora   | citeseer | pubmed  |   ACM   | ogbn-arxiv |
| :-----------------------------------------: | :--------: | :---------: | :-----: | :-----: | :------: | :-----: | :-----: | :--------: |
| [@miziha-zp](https://github.com/miziha-zp/) |    CoLA    |   0.7854/   | 0.7513/ | 0.8779/ | 0.8968/  | 0.9512/ | 0.8237/ |  0.8073/   |
|               @sjk                              |   SL-GAD   |   0.8184/   | 0.7966/ | 0.9130/ | 0.9136/  | 0.9672/ | 0.8538/ |     /      |
|               @gzn                              |  ANEMONE   |      /      |    /    | 0.9057/ | 0.9189/  | 0.9548/ |    /    |     /      |
|  [@GavinYGM](https://github.com/GavinYGM/)  |  DOMINANT  |   0.7813/0.5701   | 0.7490/0.5475 |    /0.9554    |    /0.8455     |    /oom    | 0.7494/oom |     /oom      |
|  [@GavinYGM](https://github.com/GavinYGM/)  |   ComGA    |   0.814/    | 0.799/  | 0.884/  | 0.9167/  | 0.922/  | 0.8496/ |     /      |
|  [@GavinYGM](https://github.com/GavinYGM/)   | AnomalyDAE |      0.9781/      |    0.9722/    |    /    |    /     |    /    |    0.9005/    |     /      |
|      [@Xinstein-rx](https://github.com/Xinstein-rx)                                       |   ALARM    |      /      |    /    |    /    |    /     |    /    |    /    |     /      |
| [@fmc123653](https://github.com/fmc123653/) |  AAGNN   |   0.8184/   | 0.8299/ |    /    |    /     | 0.8564/ |    /    |     /      |


## Citation