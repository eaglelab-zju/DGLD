Examples
-------
```python
>>> gnd_dataset = GraphNodeAnomalyDectionDataset("Cora", p = 15, k = 50)
>>> g = gnd_dataset[0]
>>> label = gnd_dataset.anomaly_label
>>> model = GUIDE(g.ndata['feat'].shape[1],256,6,64,num_layers=4,dropout=0.6)
>>> model.fit(g,lr=0.001,num_epoch=200,device='0',alpha=0.9986,verbose=True,y_true=label)
>>> result = model.predict(g,alpha=0.9986)
>>> print(split_auc(label, result))
```

|Method|Cora|Citrseer|Pubmed|BlogCatalog|Flickr|ACM|Arxiv|
|----|----|----|----|----|----|----|----|
|Guide|0.9815|0.9770|0.9452|0.7668| 0.7331 |0.7100| 0.7711 |