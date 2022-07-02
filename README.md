
<h1 align="center">
    <p>A Deep Graph Anomaly Detection Library based on DGL</p>
</h1>

DGLD is an open-source library for Deep Graph Anomaly Detection based on pytorch and DGL. It provides unified interface of popular graph anomaly detection methods, including the data loader, data augmentation, model training and evaluation. Also, the widely used modules are well organized so that developers and researchers can quickly implement their own designed models.

## Overview of Library
@ZS

## Quick Start
@SJK 
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
@YGM 这部分后面换成表格，而不是图片
<img src="http://latex.codecogs.com/svg.latex?\begin{array}{c|c|c|c|c}\hline&space;\text&space;{&space;Dataset&space;}&space;&&space;\sharp&space;\text&space;{&space;nodes&space;}&space;&&space;\sharp&space;\text&space;{&space;edges&space;}&space;&&space;\sharp&space;\text&space;{&space;attributes&space;}&space;&&space;\sharp&space;\text&space;{&space;anomalies&space;}&space;\\\hline&space;\text&space;{&space;BlogCatalog&space;}&space;&&space;5,196&space;&&space;171,743&space;&&space;8,189&space;&&space;300&space;\\\text&space;{&space;Flickr&space;}&space;&&space;7,575&space;&&space;239,738&space;&&space;12,407&space;&&space;450&space;\\\text&space;{&space;ACM&space;}&space;&&space;16,484&space;&&space;71,980&space;&&space;8,337&space;&&space;600&space;\\\text&space;{&space;Cora&space;}&space;&&space;2,708&space;&&space;5,429&space;&&space;1,433&space;&&space;150&space;\\\text&space;{&space;Citeseer&space;}&space;&&space;3,327&space;&&space;4,732&space;&&space;3,703&space;&&space;150&space;\\\text&space;{&space;Pubmed&space;}&space;&&space;19,717&space;&&space;44,338&space;&&space;500&space;&&space;600&space;\\\text&space;{&space;ogbn-arxiv&space;}&space;&&space;169,343&space;&&space;1,166,243&space;&&space;128&space;&&space;6000&space;\\\hline\end{array}" title="http://latex.codecogs.com/svg.latex?\begin{array}{c|c|c|c|c}\hline \text { Dataset } & \sharp \text { nodes } & \sharp \text { edges } & \sharp \text { attributes } & \sharp \text { anomalies } \\\hline \text { BlogCatalog } & 5,196 & 171,743 & 8,189 & 300 \\\text { Flickr } & 7,575 & 239,738 & 12,407 & 450 \\\text { ACM } & 16,484 & 71,980 & 8,337 & 600 \\\text { Cora } & 2,708 & 5,429 & 1,433 & 150 \\\text { Citeseer } & 3,327 & 4,732 & 3,703 & 150 \\\text { Pubmed } & 19,717 & 44,338 & 500 & 600 \\\text { ogbn-arxiv } & 169,343 & 1,166,243 & 128 & 6000 \\\hline\end{array}" />


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