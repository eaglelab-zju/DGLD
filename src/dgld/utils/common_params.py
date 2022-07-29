P = 15                      #anomaly injection hyperparameter follow CoLA, for structural anomaly
K = 50                      #anomaly injection hyperparameter follow CoLA, for contextual anomaly
Q_MAP = {                   #anomaly injection hyperparameter follow CoLA, for structural anomaly
    "Cora": 5,
    "Citeseer": 5,
    "Pubmed": 20,
    "BlogCatalog": 10,
    "Flickr": 15,
    "ogbn-arxiv": 200,
}

IN_FEATURE_MAP = {
    "Cora":1433,
    "Citeseer":3703,
    "Pubmed":500,
    "BlogCatalog":8189,
    "Flickr":12047,
    "ACM":8337,
    "ogbn-arxiv":128,
}

NUM_NODES_MAP={
    "Cora":2708,
    "Citeseer":3327,
    "Pubmed":19717,
    "BlogCatalog":5196,
    "Flickr":7575,
    "ACM":16484,
    "ogbn-arxiv":169343,
}


