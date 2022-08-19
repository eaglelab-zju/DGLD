<p align="center">
    <a href="https://zhoushengisnoob.github.io/projects/DGLD_Fronted/index.html"> <img src="DGLD_logo.jpg" width="200"/></a>
<p>

<h1 align="center">
    <p>A Deep Graph Anomaly Detection Library <br> based on DGL</p>
</h1>

<p align="center">
    <b> <a href="https://zhoushengisnoob.github.io/projects/DGLD_Fronted/index.html">Website</a> | <a href="https://zhoushengisnoob.github.io/DGLD/doc/docstring_html/html/dgld.html">Doc</a> </b>
</p>

DGLD is an open-source library for Deep Graph Anomaly Detection based on pytorch and DGL. It provides unified interface of popular graph anomaly detection methods, including the data loader, data augmentation, model training and evaluation. Also, the widely used modules are well organized so that developers and researchers can quickly implement their own designed models. 


## News
* [Aug 2022] We have released an [easy-to-use graphical command line tool](https://zhoushengisnoob.github.io/DGLD/web/index.html) for users to run experiments with different models, datasets and customized parameters. Users can select all the settings in the page, click 'Submit' and copy the shell scripts to the terminal. 
* [July 2022] For PyG users, we recommend the [PyGOD](https://github.com/pygod-team/pygod/), which is another comprehensive package that also supports many graph anomaly detection methods.
* [June 2022] Recently we receive feedback that the reported results are slightly different from the original paper. This is due to the anomaly injection setting, the graph augmentation and sampling. We will provide more details on the settings. 

## Installation
Basic environment installation:
```shell
conda create -n dgld python=3.8.0
conda activate dgld
conda install cudatoolkit==11.3.1
pip install dgl-cu113==0.8.1 dglgo==0.0.1 -f https://data.dgl.ai/wheels/repo.html
pip install torch==1.11.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```
Then clone the DGLD project, enter the directory and run:
```shell
pip install -r requirements.txt
```
To check whether you have successfully installed the package and environment, you can simply run
```shell
python example.py
```
Now you can enjoy DGLD!

## Quick Start

We support an example.py showing how it works. Here, we introduce how to simply run DGLD, following 4 steps.

### Dataloader

DGLD support multiple data import methods, including [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [DGL](https://www.dgl.ai/) and custom data. DGLD combines the process of data load and anomaly injection. Except for some basic datasets(including "Cora", "Citeseer", "Pubmed", "BlogCatalog", "Flickr", "ogbn-arxiv" and "ACM"), DGLD also accept custom data.

### Anomaly Injection

In anomaly detection, DGLD inject the abnormal node in two methods, structural and contextual, by two parameters - p and k. gnd_dataset is an instance of GraphNodeAnomalyDectionDataset. g is an instance of DGL.Graph. label is an instnace of torch.Tensor, presenting the anomaly class. Following is an example showing that a few lines of codes are sufficient to load and inject.

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



## Implemented Results ([Sorted Results](https://zhoushengisnoob.github.io/projects/DGLD_Fronted/leaderboard.html))
|                                  Method                                   |  Cora  | Citeseer | Pubmed | BlogCatalog | Flickr |  ACM   | Arxiv  |
|:-------------------------------------------------------------------------:|:------:|:--------:|:------:|:-----------:|:------:|:------:|:------:|
|                 [CoLA](https://arxiv.org/abs/2103.00113)                  | 0.8823 |  0.8765  | 0.9632 |   0.6488    | 0.5790 | 0.8194 | 0.8833 |
| [SL-GAD](https://arxiv.org/pdf/2108.09896.pdf?ref=https://githubhelp.com) | 0.8937 |  0.9003  | 0.9532 |   0.7782    | 0.7664 | 0.8146 | 0.7483 |
|       [ANEMONE](https://dl.acm.org/doi/abs/10.1145/3459637.3482057)       | 0.8916 |  0.8633  | 0.9630 |      -      |   -    |   -    |   -    |
|   [DOMINANT](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67)   | 0.8555 |  0.8236  | 0.8295 |   0.7795    | 0.7559 | 0.7067 |   -    |
|        [ComGA](https://dl.acm.org/doi/abs/10.1145/3488560.3498389)        | 0.9677 |  0.8020  | 0.9205 |   0.7908    | 0.7346 | 0.7147 |   -    |
|            [AnomalyDAE](https://arxiv.org/pdf/2002.03665.pdf)             | 0.9679 |  0.8832  | 0.9182 |   0.7666    | 0.7437 | 0.7091 |   -    |
|      [ALARM](https://ieeexplore.ieee.org/abstract/document/9162509)       | 0.9479 |  0.8318  | 0.8296 |   0.7718    | 0.7596 | 0.6952 |   -    |
| [AAGNN](https://www4.comp.polyu.edu.hk/~xiaohuang/docs/Shuang_CIKM21.pdf) | 0.7371 |  0.7616  | 0.7442 |   0.7648    | 0.7388 | 0.4868 |   -    |
|           [GUIDE](https://ieeexplore.ieee.org/document/9671990)           | 0.9785 |  0.9778  | 0.9535 |   0.7675    | 0.7337 | 0.7153 |   -    |
|  [CONAD](https://link.springer.com/chapter/10.1007/978-3-031-05936-0_35)  | 0.9646 |  0.9116  | 0.9396 |   0.7863    | 0.7395 | 0.7005 | 0.6365 |
|        [GAAN](https://dl.acm.org/doi/abs/10.1145/3340531.3412070)         | 0.7964 |  0.7979  | 0.7862 |   0.7320    | 0.7510 | - | 0.8605 |
|        [DONE](https://dl.acm.org/doi/abs/10.1145/3336191.3371788)         | 0.9636 |  0.8948  | 0.8803 |   0.7842    | 0.7555 | 0.7094 | 0.7093 |
|       [ONE](https://ojs.aaai.org/index.php/AAAI/article/view/3763)        | 0.9717 |  0.9900  | 0.8991 |   0.7924    | 0.7712 | 0.7072 |   -    |
|        [AdONE](https://dl.acm.org/doi/abs/10.1145/3336191.3371788)        | 0.9629 |  0.8935  | 0.9030 |   0.7438    | 0.7595 |   -    | 0.7651 |
|                 [GCNAE](https://arxiv.org/abs/1611.07308)                 | 0.7707 |  0.7696  | 0.7941 |   0.7363    | 0.7529 |   -    | 0.7530 |
|          [MLPAE](https://dl.acm.org/doi/10.1145/2689746.2689747)          | 0.7617 |  0.7538  | 0.7211 |   0.7399    | 0.7514 |   -    | 0.7382 |
|                                 [SCAN](https://dl.acm.org/doi/10.1145/1281192.1281280)                                  | 0.6508 |  0.6671  | 0.7361 |   0.4926    | 0.6498 |   -    | 0.6905 |

## Upcoming Features
* More Graph Anomaly Detection Methods
* Edge/Community/Graph Level Anomaly Detection Tasks
* Graphical Operation Interface
