import torch 
import dgl 
import torch.nn as nn 
import torch.nn.functional as F 
from .anomalous_utils import Laplacian
from utils.early_stopping import EarlyStopping
import numpy as np 


class ANOMALOUS_model(nn.Module):
    def __init__(self,w,r):
        super().__init__()
        self.W = nn.Parameter(w)
        self.R = nn.Parameter(r)
    def forward(self,x):
        return x @ self.W @ x,self.R


class ANOMALOUS(nn.Module):
    def __init__(self):
        super().__init__()
        
    def fit(self, graph:dgl.DGLGraph,num_epoch,gamma=0.01,weight_decay=0.01,lr = 0.004,device = 'cpu'):
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')   
        L = Laplacian(graph)
        x = graph.ndata['feat']
        x = F.normalize(x,dim=0)
        w_init = torch.randn_like(x.T)
        r_init = torch.inverse((1+weight_decay)*torch.eye(x.shape[0]) + gamma * L) @ x
        self.model = ANOMALOUS_model(w_init,r_init)
        L = L.to(device)
        x = x.to(device)
        self.model = self.model.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(),lr=lr,weight_decay=weight_decay)

        
        early_stop = EarlyStopping(early_stopping_rounds=100,patience=5,check_finite=False)
        for epoch in range(num_epoch):
            x_, r = self.model(x)
            loss =  torch.norm(x - x_ - r, 2) + gamma * torch.trace(r.T @ L @ r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            early_stop(loss)
            if early_stop.isEarlyStopping():
                print(f'early stop in round {epoch}')
                break 

    def predict(self,graph):
        return -torch.sum(torch.pow(self.model.R, 2), dim=1).detach().cpu().numpy()



        





