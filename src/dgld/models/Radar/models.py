from pickletools import optimize
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from dgld.utils.early_stopping import EarlyStopping
from dgld.utils.common import preprocess_features

class Radar():
    def __init__(self):
        pass
    
    def fit(self,
            graph:dgl.DGLGraph,
            alpha:float,
            beta:float,
            gamma:float,
            device='cpu',
            num_epoch=1,
            eps=0.,
            verbose=True,
            ):
        print('*'*20,'training','*'*20)
        
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!')
            
        graph = graph.remove_self_loop()
        num_nodes = graph.number_of_nodes()
        
        A = graph.adj().to_dense()
        A = torch.max(A, A.T)
        X = graph.ndata['feat']
        X = preprocess_features(X)
        # Build Laplacian matrix L from the adjacency matrix A
        D = torch.diag(torch.sum(A, 1))
        L = D - A
        # Initialize D_R and D_W to be identity matrix
        Dr = torch.eye(num_nodes)
        Dw = torch.eye(num_nodes)
        # Initialize R = (I + βDR + γL)−1X
        I = torch.eye(num_nodes)
        R = torch.inverse(I + beta * Dr + gamma * L) @ X
        
        map(lambda x: x.to(device), [A, X, D, L, Dr, Dw, I, R])
        
        early_stop = EarlyStopping(early_stopping_rounds=10, patience=1, delta=0.001, verbose=True)
        
        for epoch in range(num_epoch):
            # update w
            W = torch.inverse(X @ X.T + alpha * Dw) @ (X @ X.T - X @ R.T)
            Dw = torch.diag(1. / (2. * torch.norm(W, dim=1) + eps))
            # update r
            R = torch.inverse(I + beta * Dr + gamma * L) @ (X - W.T @ X)
            Dr = torch.diag(1. / (2. * torch.norm(R, dim=1) + eps))

            loss = torch.norm(X - W.T @ X - R) + \
                alpha * torch.sum(torch.norm(W, dim=1)) + \
                beta * torch.sum(torch.norm(R, dim=1)) + \
                gamma * torch.trace(R.T @ L @ R)
            
            if verbose is True:    
                print(f"Epoch: {epoch:04d}, train/loss={loss:.5f}")
            
            early_stop(loss)
            if early_stop.isEarlyStopping():
                print(f"Early stopping in round {epoch}")
                break
        
        self.R = R
        
        # self.model = Radar_Base(alpha, beta, gamma)
        # self.model.reset_parameters(A, X)
        # self.model.W = nn.Parameter(W)
        # self.model.R = nn.Parameter(R)
        
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        
        # for epoch in range(num_epoch):
        #     tmp, r = self.model(X)
        #     loss = self.model.loss(X)
            
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
            
        #     if verbose is True:    
        #         print(f"Epoch: {epoch:04d}, train/loss={loss.item():.5f}")
        
        # self.R = self.model.R.cpu().detach() 
    
    def predict(self, graph, device='cpu'):
        print('*'*20,'predict','*'*20)
        
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!')
        else:
            device = torch.device("cpu")
            print('Using cpu!!!') 
        
        predict_score = torch.norm(self.R, dim=1)
        predict_score = predict_score.cpu().detach().numpy()
        
        return predict_score
    
    
class Radar_Base(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(Radar_Base, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def reset_parameters(self, A, X):
        num_nodes = A.shape[0]
        # Build Laplacian matrix L from the adjacency matrix A
        D = torch.diag(torch.sum(A, 1))
        self.L = D - A
        # Initialize D_R and D_W to be identity matrix
        Dr = torch.eye(num_nodes)
        Dw = torch.eye(num_nodes)
        # Initialize R = (I + βDR + γL)−1X
        I = torch.eye(num_nodes)
        self.R = nn.Parameter(torch.inverse(I + self.beta * Dr + self.gamma * self.L) @ X)
        self.W = nn.Parameter(torch.eye(num_nodes))
        
    def forward(self, X):
        return self.W.T @ X, self.R
    
    def loss(self, X):
        loss = torch.norm(X - self.W.T @ X - self.R) + \
                self.alpha * torch.sum(torch.norm(self.W, dim=1)) + \
                self.beta * torch.sum(torch.norm(self.R, dim=1)) + \
                self.gamma * torch.trace(self.R.T @ self.L @ self.R)
                
        return loss