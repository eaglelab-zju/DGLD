import torch 
import numpy as np 
import dgl 
import torch.nn as nn 
import torch.nn.functional as F 
from sklearn.metrics import roc_auc_score
import os,sys
current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))) + '/utils/'
sys.path.append(current_dir)
from utils.evaluation import split_auc
class GAAN():
    """
    GAAN (Generative Adversarial Attributed Network Anomaly
    Detection) is a generative adversarial attribute network anomaly
    detection framework, including a generator module, an encoder
    module, a discriminator module, and uses anomaly evaluation
    measures that consider sample reconstruction error and real sample
    recognition confidence to make predictions.

    Parameters
    ----------
    noise_dim : int
        Dimension of the Gaussian random noise
    gen_hid_dims : List
        The list for the size of each hidden sample for generator.
    attrb_dim : int
        The attribute of node's dimension
    ed_hid_dims : List
        The list for the size of each hidden sample for encoder.
    out_dim : int
        Dimension of encoder output.
    dropout : int, optional
        Dropout probability of each hidden, default 0
    act : callable activation function, optional
        The non-linear activation function to use, default torch.nn.functional.relu

    Examples
    -------
    >>> gnd_dataset = GraphNodeAnomalyDectionDataset("Cora")
    >>> g = gnd_dataset[0]
    >>> label = gnd_dataset.anomaly_label
    >>> model = GAAN(32,[32,64,128],g.ndata['feat'].shape[1],[32,64],128,dropout = 0.2)
    >>> model.fit(g, num_epoch=1, device='cpu')
    >>> result = model.predict(g)
    >>> print(split_auc(label, result))
    """
    def __init__(self,noise_dim,gen_hid_dims,attrb_dim,ed_hid_dims,out_dim,dropout=0,act=F.relu):
        self.model = GAAN_model(noise_dim,gen_hid_dims,attrb_dim,ed_hid_dims,out_dim,dropout=dropout,act=act)
        self.noise_dim = noise_dim

    def fit(self,graph,attrb_feat=None,batch_size=0,num_epoch=10,g_lr=0.001,d_lr=0.001,
            weight_decay=0,num_neighbor = -1,device='cpu',verbose=True,y_true=None,alpha=0.3):
        """
        Train the model.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.
        attrb_feat : torch.tensor, optional
            The attritube matrix of nodes. None for use the graph.ndata['feat'], default None
        batch_size : int, optional
            Minibatch size, 0 for full batch, default 0
        num_epoch : int, optional
            _description_, by default 10
        g_lr : float, optional
            Generator learning rate, default 0.001
        d_lr : float, optional
            Discriminator learning, by default 0.001
        weight_decay : int, optional
            Weight decay (L2 penalty), by default 0
        num_neighbor : int, optional
            The number of the simple number, -1 for all neighber, default -1
        device : str, optional
            device, default 'cpu'
        verbose : bool, optional
            Verbosity mode, default False
        y_true : torch.tensor, optional
            The optional outlier ground truth labels used to monitor the training progress, by default None
        alpha : float, optional
            Loss balance weight for attribute and structure, default 0.3
        """
        print('*'*20,'training','*'*20)
        if attrb_feat is None:
            attrb_feat = graph.ndata['feat']
        if batch_size == 0:
            batch_size = graph.num_nodes()
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')  
        graph = dgl.to_simple(graph)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        self.model.to(device)

        opt_g = torch.optim.Adam(self.model.generator.parameters(),g_lr,weight_decay=weight_decay)
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(),d_lr,weight_decay=weight_decay)

        sampler = dgl.dataloading.NeighborSampler([num_neighbor])
        nid = torch.arange(graph.num_nodes())

        dataloader = dgl.dataloading.DataLoader(graph, nid, sampler,batch_size=batch_size,
        shuffle=True, drop_last=False)
        
        self.model.train()
        for epoch in range(num_epoch):
            # print(list(self.model.generator.parameters()))
            epoch_loss_d = 0
            epoch_loss_g = 0
            score = torch.zeros(graph.num_nodes())

            for input_nodes, output_nodes, blocks in dataloader:
                src_node = blocks[0].edges()[0]
                dst_node = blocks[0].edges()[1]
                
                x = attrb_feat[input_nodes].to(device)

                # train discriminator
                random_noise = torch.randn(x.shape[0], self.noise_dim).to(device)
                x_,a,a_ = self.model(random_noise,x)
                loss_d = self.dis_loss(a[src_node,dst_node],a_[src_node,dst_node].detach())
                opt_d.zero_grad()
                loss_d.backward()
                nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), 0.001)
                opt_d.step()
                epoch_loss_d += loss_d.item() * len(output_nodes)

                # train generator 
                random_noise = torch.randn(x.shape[0], self.noise_dim).to(device)
                x_,a,a_ = self.model(random_noise,x)
                loss_g = self.gen_loss(a_[src_node,dst_node])
                opt_g.zero_grad()
                loss_g.backward()
                # nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), 0.001)
                opt_g.step()
                epoch_loss_g += loss_g.item() * len(output_nodes)

                if verbose and y_true is not None:
                    subgraph = graph.subgraph(input_nodes,relabel_nodes=True)
                    score[output_nodes] = self.cal_score(subgraph,x,x_,a,alpha,len(output_nodes)).detach().cpu()
            epoch_loss_d /= graph.num_nodes()
            epoch_loss_g /= graph.num_nodes()
            if verbose:
                print("Epoch:", '%04d' % (epoch), "generator_loss=", "{:.5f}".format(epoch_loss_g),
                    "discriminator_loss=", "{:.5f}".format(epoch_loss_d))
                if y_true is not None:  
                    split_auc(y_true, score)

    def predict(self,graph,attrb_feat=None,alpha = 0.3,batch_size = 0,device='cpu',num_neighbor = -1):
        """
        Test model

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.
        attrb_feat : torch.tensor, optional
            The attritube matrix of nodes. None for use the graph.ndata['feat'], default None
        alpha : float, optional
            Loss balance weight for attribute and structure, default 0.3
        batch_size : int, optional
            Minibatch size, 0 for full batch, default 0
        device : str, optional
            device, default 'cpu'
        num_neighbor : int, optional
            The number of the simple number, -1 for all neighber, default -1

        Returns
        -------
        torch.tensor
            The score of all nodes.
        """
        print('*'*20,'predict','*'*20)
        if attrb_feat is None:
            attrb_feat = graph.ndata['feat']
        if batch_size == 0:
            batch_size = graph.num_nodes()

        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!') 
        graph = dgl.to_simple(graph)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        self.model = self.model.to(device)


        sampler = dgl.dataloading.NeighborSampler([num_neighbor])
        nid = torch.arange(graph.num_nodes())

        dataloader = dgl.dataloading.DataLoader(graph, nid, sampler,batch_size=batch_size,
        shuffle=False, drop_last=False)

        self.model.eval()
        score = torch.zeros(graph.num_nodes())

        for input_nodes, output_nodes, blocks in dataloader:
            x = attrb_feat[input_nodes].to(device)
            random_noise = torch.randn(x.shape[0], self.noise_dim).to(device)
            x_,a,a_ = self.model(random_noise,x)
            subgraph = graph.subgraph(input_nodes,relabel_nodes=True)
            score[output_nodes] = self.cal_score(subgraph,x,x_,a,alpha,len(output_nodes)).detach().cpu()
        return score 

    def cal_score(self,graph,x,x_,a,alpha,batch_size):
        """
        The function to compute score.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph.
        x : torch.tensor
            The attritube matrix.
        x_ : torch.tensor
            The generator attritube matrix.
        a : torch.tensor
            The reconstruction adjacency matrix by real attritube.
        alpha : float
            Loss balance weight for attribute and structure. 
        batch_size : int
            The number of nodes to compute.

        Returns
        -------
        torch.tensor
            The score of nodes.
        """
        diff_score = torch.pow(x[:batch_size]-x_[:batch_size],2)
        L_g = torch.sqrt(torch.mean(diff_score,dim=1))

        adj = graph.adj().to_dense()[:batch_size].to(a.device)
        a = a[:batch_size]

        a = F.binary_cross_entropy(a, torch.ones_like(a), reduction='none')
        L_d = torch.sum(adj * a,dim=1)/(torch.sum(adj,dim=1)+1)
        return alpha * L_g + (1 - alpha) * L_d 


    def gen_loss(self,a_):
        """
        The function to compute generator loss.

        Parameters
        ----------
        a_ : torch.tensor
            The probability of edge from the fake attribute reconstruction adjacency matrix.

        Returns
        -------
        torch.tensor
            Generator loss.
        """
        loss_g = F.binary_cross_entropy(a_, torch.ones_like(a_), reduction='mean') 
        return loss_g
     
    def dis_loss(self,a,a_):
        """
        The function to compute discriminator loss.

        Parameters
        ----------
        a : torch.tensor
            The probability of edge from the true attribute reconstruction adjacency matrix.
        a_ : torch.tensor
            The probability of edge from the fake attribute reconstruction adjacency matrix.

        Returns
        -------
        torch.tensor
            Discriminator loss.
        """
        loss_d = F.binary_cross_entropy(a, torch.ones_like(a), reduction='mean') 
        loss_d_ = F.binary_cross_entropy(a_, torch.zeros_like(a_), reduction='mean') 
        return (loss_d + loss_d_)/2

class MLP(nn.Module):
    """
    MLP model for  generator and encoder

    Parameters
    ----------
    in_channels : int
        Size of each input sample
    hid_layers : List
        The list for the size of each hidden sample.
    out_channels : int
        Size of each output sample.
    dropout : int, optional
        Dropout probability of each hidden, default 0
    act : callable activation function, optional
        The non-linear activation function to use, default torch.nn.functional.relu
    """
    def __init__(self,in_channels,hid_layers,out_channels,dropout=0,act= F.relu,batch_norm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels,hid_layers[0]))
        self.norms = nn.ModuleList()
        # self.norms.append(nn.BatchNorm1d(in_channels))
        for i in range(len(hid_layers)-1):
            if batch_norm:
                self.norms.append(nn.BatchNorm1d(hid_layers[i]))
            else:
                self.norms.append(nn.Identity())
            self.layers.append(nn.Linear(hid_layers[i],hid_layers[i+1]))
        if batch_norm:
            self.norms.append(nn.BatchNorm1d(hid_layers[-1]))
        else:
            self.norms.append(nn.Identity())
        self.layers.append(nn.Linear(hid_layers[-1],out_channels))
        self.dropout = nn.Dropout(dropout)
        self.act = act 
    def forward(self,in_feat):
        """
        The function to compute forward of MLP

        Parameters
        ----------
        in_feat : torch.tensor
            The feature of the input data

        Returns
        -------
        torch.tensor
            The output of MLP
        """
        in_feat = self.layers[0](in_feat)
        for layer,norm in zip(self.layers[1:],self.norms):
            # in_feat = norm(in_feat)
            in_feat = self.act(in_feat)
            in_feat = norm(in_feat)
            in_feat = self.dropout(in_feat)
            in_feat = layer(in_feat)
        
        return in_feat
         
        

class GAAN_model(nn.Module):
    """
    GAAN base model

    Parameters
    ----------
    noise_dim : int
        Dimension of the Gaussian random noise
    gen_hid_dims : List
        The list for the size of each hidden sample for generator.
    attrb_dim : int
        The attribute of node's dimension
    ed_hid_dims : List
        The list for the size of each hidden sample for encoder.
    out_dim : int
        Dimension of encoder output.
    dropout : int, optional
        Dropout probability of each hidden, default 0
    act : callable activation function, optional
        The non-linear activation function to use, default torch.nn.functional.relu
    """
    def __init__(self,noise_dim,gen_hid_dims,attrb_dim,ed_hid_dims,out_dim,dropout=0,act=F.relu):
        
        super().__init__()
        self.generator = MLP(noise_dim,gen_hid_dims,attrb_dim,dropout=dropout,act=act)
        self.discriminator = MLP(attrb_dim,ed_hid_dims,out_dim,dropout=dropout,act=act,batch_norm=False)

    def forward(self,random_noise,x):
        """
        The function to compute forward.

        Parameters
        ----------
        random_noise : torch.tensor
            The random noise.
        x : torch.tensor
            The ture attritube for node.

        Returns
        -------
        x_: torch.tensor
            The generated attritube
        a: torch.tensor
            The reconstruction adjacency matrix by real attritube.
        a_: torch.tensor
            The reconstruction adjacency matrix by fake attritube.
        """
        x_ = self.generator(random_noise)

        z = self.discriminator(x)
        a = torch.sigmoid(torch.mm(z,z.T))

        z_ = self.discriminator(x_)
        a_ = torch.sigmoid(torch.mm(z_,z_.T))
        return x_,a,a_