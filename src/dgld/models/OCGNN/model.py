import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch import GraphConv, GATConv
from dgl.nn.pytorch.conv import SAGEConv
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from torch.utils.tensorboard import SummaryWriter


class GCN(nn.Module):
    """
        Base GCN model
    """
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden*2, bias=False, activation=activation))
        # hidden layers
        for i in range(1, n_layers-1):
            self.layers.append(GraphConv(n_hidden*2, n_hidden*2, bias=False, activation=activation))
        self.layers.append(GraphConv(n_hidden*2, n_hidden, bias=False))
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation

    def forward(self, g, features):
        h = features
        if isinstance(g, list):
            # mini batch forward compute
            assert len(g) == len(self.layers)
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(g[i], h)
        else:
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(g, h)

        return h


class GraphSAGE(nn.Module):
    """
        Base GraphSAGE model
    """
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden * 2, aggregator_type, feat_drop=dropout, bias=True, activation=activation))
        # hidden layers
        for i in range(1, n_layers-1):
            self.layers.append(SAGEConv(n_hidden * 2, n_hidden * 2, aggregator_type, feat_drop=dropout, bias=True, activation=activation))
        self.layers.append(SAGEConv(n_hidden * 2, n_hidden, aggregator_type, feat_drop=dropout, bias=True, activation=None))
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation

    def forward(self, g, features):
        h = features
        if isinstance(g, list):
            # mini batch forward compute
            assert len(g) == len(self.layers)
            for i, layer in enumerate(self.layers):
                h = layer(g[i], h)
        else:
            for i, layer in enumerate(self.layers):
                h = layer(g, h)

        return h


class GAT(nn.Module):
    """
        Base GAT model
    """
    def __init__(self,
                 n_layers,
                 in_feats,
                 n_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.n_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if n_layers > 1:
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_feats, n_hidden * 2, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for i in range(1, n_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    n_hidden * 2 * heads[i-1], n_hidden * 2, heads[i],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                n_hidden * 2 * heads[-2], n_hidden, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GATConv(
                in_feats, n_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, features):
        h = features
        if isinstance(g, list):
            # mini batch forward compute
            assert len(g) == len(self.layers)
            for i, layer in enumerate(self.gat_layers):
                h = layer(g[i], h)
                h = h.flatten(1) if i != self.n_layers - 1 else h.mean(1)
        else:
            for i, layer in enumerate(self.gat_layers):
                h = layer(g, h)
                h = h.flatten(1) if i != self.n_layers - 1 else h.mean(1)

        return h


class OCGNNModel(nn.Module):
    """
    OCGNN base model

    Parameters
    ----------
    feat_size : int
        dimension of input node feature.
    module : str, optional
        type of GNN kernel. Defaults: 'GCN'.
    hidden_dim : int, optional
        dimension of hidden layers' feature. Defaults: 128.
    n_layers : int, optional
        number of network layers. Defaults: 2.
    dropout : float, optional
        dropout probability. Defaults: 0.5.
    act : callable activation function, optional
        Activation function. Default: torch.nn.functional.relu.

    """
    def __init__(self,
                 feat_size,
                 module='GCN',
                 hidden_dim=128,
                 n_layers=2,
                 dropout=0.5,
                 act=F.relu
                 ):
        super(OCGNNModel, self).__init__()
        self.net = None
        if module == 'GCN':
            self.net = GCN(feat_size,
                           hidden_dim,
                           n_layers,
                           act,
                           dropout)
        if module == 'GraphSAGE':
            self.net = GraphSAGE(feat_size,
                                 hidden_dim,
                                 n_layers,
                                 act,
                                 dropout,
                                 aggregator_type='pool')
        if module == 'GAT':
            self.net = GAT(n_layers,
                           feat_size,
                           hidden_dim,
                           heads=([8] * (n_layers-1)) + [1],
                           activation=act,
                           feat_drop=dropout,
                           attn_drop=dropout,
                           negative_slope=0.2,
                           residual=False)

    def forward(self, blocks, h):
        """
        The function to compute forward

        Parameters
        ----------
        blocks : list
            list of DGLBlock.
        h : torch.tensor
            node feature.

        Returns
        -------
        x: torch.tensor
           node representation.

        """
        x = self.net(blocks, h)
        return x


class OCGNN(nn.Module):
    """One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks.
    [Neural Computing and Applications 2021]
    ref:https://github.com/WangXuhongCN/OCGNN

    Parameters
    ----------
    feat_size : int
        dimension of input node feature.
    module : str, optional
        type of GNN kernel. Defaults: 'GCN'.
    hidden_dim : int, optional
        dimension of hidden layers' feature. Defaults: 128.
    n_layers : int, optional
        number of network layers. Defaults: 2.
    dropout : float, optional
        dropout probability. Defaults: 0.5.
    nu : float, optional
        Regularization parameter. Defaults: 0.4.
    act : callable activation function, optional
        Activation function. Default: torch.nn.functional.relu.

    Examples
    -------
    >>> from dgld.models.OCGNN import OCGNN
    >>> model = OCGNN(feat_size=1433)
    >>> model.fit(g, num_epoch=1)
    >>> result = model.predict(g)
    """
    def __init__(self, feat_size, module, hidden_dim=128, n_layers=2, dropout=0.5, nu=0.4, act=F.relu):
        super(OCGNN, self).__init__()
        self.n_layers = n_layers
        self.eps = 0.001
        self.nu = nu
        self.data_center = 0
        self.radius = 0.0
        self.model = OCGNNModel(feat_size, module, hidden_dim, n_layers, dropout, act)

    def init_center(self, g, features):
        """
        Initialize hypersphere center c as the mean from
        an initial forward pass on the data.

        Parameters
        ----------
        g : DGL.Graph
            input graph.
        features : torch.Tensor
            node feature.

        Returns
        ----------
        c : torch.Tensor
            The new centroid.
        """
        n_samples = 0
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(g, features)
            # get the inputs of the batch
            n_samples = outputs.shape[0]
            c = torch.sum(outputs, dim=0)
        # print(outputs)
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be
        # trivially matched with zero weights.
        c[(abs(c) < self.eps) & (c < 0)] = -self.eps
        c[(abs(c) < self.eps) & (c > 0)] = self.eps

        return c

    def get_radius(self, dist):
        """
        Optimally solve for radius R via the (1-nu)-quantile of distances.

        Parameters
        ----------
        dist : torch.Tensor
            Distance of the data points, calculated by the loss function.

        Returns
        ----------
        radius : numpy.array
            New radius.
        """
        radius = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()),
                             1 - self.nu)
        return radius

    def anomaly_scores(self, outputs, mask=None):
        """
        Calculate the anomaly score given by Euclidean distance to the center.

        Parameters
        ----------
        outputs : torch.Tensor
            The output in the reduced space by GNN.
        mask : torch.tensor, optional
            The split mask of node data.

        Returns
        ----------
        dist : torch.Tensor
            Average distance.
        scores : torch.Tensor
            Anomaly scores.
        """
        if mask is None:
            dist = torch.sum((outputs - self.data_center) ** 2, dim=1)
        else:
            dist = torch.sum((outputs[mask] - self.data_center) ** 2, dim=1)

        scores = dist - self.radius ** 2

        return dist, scores

    def loss_function(self, outputs, mask=None):
        """
        Calculate the loss in paper Equation (4)

        Parameters
        ----------
        outputs : torch.Tensor
            The output in the reduced space by model.
        mask : torch.tensor, optional
            The split mask of node data.

        Returns
        ----------
        loss : torch.Tensor
            A combined loss of radius and average scores.
        dist : torch.Tensor
            Average distance.
        scores : torch.Tensor
            Anomaly scores.
        """
        dist, scores = self.anomaly_scores(outputs, mask)

        loss = self.radius ** 2 + (1 / self.nu) * torch.mean(
            torch.max(torch.zeros_like(scores), scores))

        return loss, dist, scores

    def fit(self,
            g,
            lr=1e-3,
            batch_size=0,
            num_epoch=1,
            warmup_epoch=3,
            log_dir='tmp',
            weight_decay=0.,
            device='cpu'
            ):
        """Fitting model

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset.
        lr : float, optional
            learning rate. Defaults: 1e-3.
        batch_size : int, optional
            the size of training batch. Defaults: 0 for full graph train.
        num_epoch : int, optional
            number of training epochs. Defaults: 1.
        warmup_epoch : int, optional
            number of warmup epochs. Defaults: 3.
        log_dir : str, optional
            log dir. Defaults: 'tmp'.
        weight_decay : float, optional
            weight decay (L2 penalty). Defaults: 0.
        device : str, optional
            device of computation. Defaults: 'cpu'.

        """
        print('*' * 20, 'training', '*' * 20)

        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')

        features = g.ndata['feat']

        self.model = self.model.to(device)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        features = features.to(device)

        writer = SummaryWriter(log_dir=log_dir)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if batch_size == 0:
            print("full graph training!!!")

            self.data_center = self.init_center(g, features).to(device)
            self.radius = torch.tensor(0, device=device)

            self.model.train()
            for epoch in range(num_epoch):
                outputs = self.model(g, features)

                train_loss, dist, _ = self.loss_function(outputs)

                if warmup_epoch is not None and epoch % warmup_epoch == 0:
                    self.data_center = self.init_center(g, features).to(device)
                    self.radius = torch.tensor(self.get_radius(dist), device=device)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                print("Epoch:", '%04d' % epoch, "train_loss=", "{:.5f}".format(train_loss.item()))

                writer.add_scalars(
                    "loss",
                    {"loss": train_loss.item()},
                    epoch,
                )
                writer.flush()

        else:
            print("mini batch training!!!")
            sampler = MultiLayerFullNeighborSampler(num_layers=self.n_layers)
            nid = torch.arange(g.num_nodes())
            nid = torch.LongTensor(nid).to(device)
            dataloader = NodeDataLoader(
                g, nid, sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            )

            self.data_center = self.init_center(g, features)
            self.radius = torch.tensor(0, device=device)

            self.model.train()
            for epoch in range(num_epoch):
                epoch_loss = 0.0
                ind = 0
                for input_nodes, output_nodes, blocks in dataloader:
                    blocks = [b.to(device) for b in blocks]
                    bfeatures = features[input_nodes]
                    boutputs = self.model(blocks, bfeatures)

                    train_loss, dist, _ = self.loss_function(boutputs)
                    epoch_loss += train_loss.item()

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    print("Epoch:", '%04d' % epoch, "batch:", '%d' % ind, "train_loss=", "{:.5f}".format(train_loss.item() * 100000))
                    ind += 1

                if warmup_epoch is not None and epoch % warmup_epoch == 0:
                    self.data_center = self.init_center(g, features).to(device)
                    self.radius = torch.tensor(self.get_radius(dist), device=device)

                print("Epoch:", '%04d' % epoch, "train_loss=", "{:.5f}".format(epoch_loss))

                writer.add_scalars(
                    "loss",
                    {"loss": train_loss.item()},
                    epoch,
                )
                writer.flush()

    def predict(self,
                g,
                batch_size=0,
                device='cpu'
                ):
        """predict and return anomaly score of each node

        Parameters
        ----------
        g : dgl.DGLGraph
            graph dataset.
        batch_size : int, optional
            the size of predict batch. Defaults: 0 for full graph predict.
        device : str, optional
            device of computation. Defaults: 'cpu'.

        Returns
        -------
        predict_score : numpy.ndarray
            anomaly score of each node.
        """
        print('*' * 20, 'predict', '*' * 20)

        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')

        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        self.model = self.model.to(device)
        self.data_center = self.data_center.to(device)
        self.radius = self.radius.to(device)
        self.model.eval()

        if batch_size == 0:
            features = g.ndata['feat'].to(device)
            with torch.no_grad():
                outputs = self.model(g, features)

            _, predict_score = self.anomaly_scores(outputs)

        else:
            sampler = MultiLayerFullNeighborSampler(self.n_layers)
            nid = torch.arange(g.num_nodes()).to(device)
            dataloader = NodeDataLoader(g, nid, sampler,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        drop_last=False
                                        )

            with torch.no_grad():
                predict_score = torch.zeros(g.num_nodes()).to(device)
                for input_nodes, output_nodes, blocks in dataloader:
                    blocks = [b.to(device) for b in blocks]
                    feat = blocks[0].srcdata['feat']
                    boutput = self.model(blocks, feat)
                    _, bscore = self.anomaly_scores(boutput)
                    predict_score[output_nodes] = bscore

        predict_score = predict_score.cpu().detach().numpy()

        return predict_score
