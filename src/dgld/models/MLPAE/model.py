""" Multilayer Perceptron Autoencoder
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from dgld.utils.early_stopping import EarlyStopping


class MLP(nn.Module):
    """
        Base MLP model
    """

    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 dropout,
                 activation
                 ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, out_feats))
        self.dropout = nn.Dropout(p=dropout)
        self.act = activation

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.act(h)
                h = self.dropout(h)
            h = layer(h)

        return h


class MLPAEModel(nn.Module):
    """This is a basic model of MLPAE.

        Parameters
        ----------
        feat_size : int
            dimension of input node feature.
        hidden_dim : int, optional
            dimension of hidden layers' feature. Defaults: 64.
        n_layers : int, optional
            number of network layers. Defaults: 2.
        dropout : float, optional
            dropout probability. Defaults: 0.3.
        act : callable activation function, optional
            Activation function. Default: torch.nn.functional.relu.

        """

    def __init__(self,
                 feat_size,
                 hidden_dim=64,
                 n_layers=2,
                 dropout=0.3,
                 act=F.relu):
        super(MLPAEModel, self).__init__()
        self.net = MLP(feat_size, hidden_dim, feat_size, n_layers, dropout, act)

    def forward(self, features):
        """Forward Propagation

        Parameters
        ----------
        features : torch.tensor
            features of nodes

        Returns
        -------
        x : torch.tensor
            Reconstructed node matrix

        """
        x = self.net(features)

        return x


class MLPAE(nn.Module):
    """ Multilayer Perceptron Autoencoder

    Parameters
    ----------
    feat_size : int
        dimension of input node feature.
    hidden_dim : int, optional
        dimension of hidden layers' feature. Defaults: 128.
    n_layers : int, optional
        number of network layers. Defaults: 2.
    dropout : float, optional
        dropout probability. Defaults: 0.3.
    act : callable activation function, optional
        Activation function. Default: torch.nn.functional.relu.

    Examples
    -------
    >>> from  dgld.models.MLPAE import MLPAE
    >>> model = MLPAE(feat_size=1433)
    >>> model.fit(g, num_epoch=1)
    >>> result = model.predict(g)
    """

    def __init__(self,
                 feat_size,
                 hidden_dim=128,
                 n_layers=2,
                 dropout=0.3,
                 act=F.relu
                 ):
        super(MLPAE, self).__init__()
        self.n_layers = n_layers
        self.model = MLPAEModel(feat_size, hidden_dim, n_layers, dropout, act=act)

    def loss_func(self, x, x_hat):
        """
        Calculate the loss

        Parameters
        ----------
        x : torch.tensor
            The original features of node data.
        x_hat : torch.tensor
            The output by model.

        Returns
        ----------
        loss : torch.tensor
            The loss of model.

        """
        loss = torch.linalg.norm(x - x_hat, dim=1)
        return loss

    def fit(self,
            g,
            lr=0.005,
            batch_size=0,
            num_epoch=100,
            weight_decay=0.,
            device=0
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

        self.model = self.model.to(device)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        features = g.ndata['feat']

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        early_stop = EarlyStopping(early_stopping_rounds=10, patience=20)

        if batch_size == 0:
            print("full graph training!!!")

            self.model.train()
            for epoch in range(num_epoch):
                x_hat = self.model(features)

                train_loss = torch.mean(self.loss_func(features, x_hat))

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                print("Epoch:", '%04d' % epoch, "train_loss=", "{:.5f}".format(train_loss.item()))

                early_stop(train_loss.cpu().detach(), self.model)
                if early_stop.isEarlyStopping():
                    print(f"Early stopping in round {epoch}")
                    break

        else:
            print("mini batch training!!!")

            sampler = MultiLayerFullNeighborSampler(num_layers=self.n_layers)
            nid = torch.arange(g.num_nodes())
            nid = torch.LongTensor(nid).to(device)
            dataloader = DataLoader(
                g, nid, sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            )

            self.model.train()
            for epoch in range(num_epoch):

                epoch_loss = 0.0

                for i, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                    bfeatures = features[input_nodes]
                    x_hat = self.model(bfeatures)

                    train_loss = torch.mean(self.loss_func(features[output_nodes], x_hat))
                    epoch_loss += train_loss.item()

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    print("Epoch:", '%04d' % epoch, "batch:", '%d' % (i + 1), "train_loss=",
                          "{:.5f}".format(train_loss.item()))

                print("Epoch:", '%04d' % epoch, "train_loss=", "{:.5f}".format(epoch_loss))

                early_stop(epoch_loss, self.model)
                if early_stop.isEarlyStopping():
                    print(f"Early stopping in round {epoch}")
                    break

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
        self.model.eval()
        features = g.ndata['feat'].to(device)

        if batch_size == 0:
            with torch.no_grad():
                x_hat = self.model(features)

            predict_score = self.loss_func(features, x_hat)

        else:
            sampler = MultiLayerFullNeighborSampler(self.n_layers)
            nid = torch.arange(g.num_nodes()).to(device)
            dataloader = DataLoader(g, nid, sampler,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        drop_last=False
                                        )

            with torch.no_grad():
                predict_score = torch.zeros(g.num_nodes()).to(device)
                for input_nodes, output_nodes, blocks in dataloader:
                    feat = features[input_nodes]
                    x_hat = self.model(feat)
                    bscore = self.loss_func(features[output_nodes], x_hat)
                    predict_score[output_nodes] = bscore

        predict_score = predict_score.cpu().detach().numpy()

        return predict_score
