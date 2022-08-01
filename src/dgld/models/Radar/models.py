import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from dgld.utils.early_stopping import EarlyStopping

class Radar():
    def __init__(self) -> None:
        pass
    
    def fit(self,
            graph:dgl.DGLGraph,
            alpha,
            beta,
            gamma,
            device='cpu',
            num_epoch=1,
            eps = 1e-6,
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
        X = graph.ndata['feat']
        # Build Laplacian matrix L from the adjacency matrix A
        D = torch.diag(graph.out_degrees())
        L = D - A
        # Initialize D_R and D_W to be identity matrix
        Dr = torch.eye(num_nodes)
        Dw = torch.eye(num_nodes)
        # Initialize R = (I + βDR + γL)−1X
        I = torch.eye(num_nodes)
        R = torch.inverse(I + beta * Dr + gamma * L) @ X
        
        [map(lambda x: x.to(device), [A, X, D, L, Dr, Dw, I, R])]
        
        early_stop = EarlyStopping(early_stopping_rounds=10, patience=10)
        
        for epoch in range(num_epoch):
            W = torch.inverse(X @ X.T + alpha * Dw) @ (X @ X.T - X @ R.T)
            Dw = torch.diag(1. / (2 * torch.norm(W, dim=1) + eps))
            R = torch.inverse(I + beta * Dr + gamma * L) @ (X - W.T @ X)
            Dr = torch.diag(1. / (2 * torch.norm(R, dim=1) + eps))

            loss = torch.norm(X - W.T @ X - R) + alpha * torch.sum(torch.norm(W, dim=1)) + beta * torch.sum(torch.norm(R, dim=1)) + gamma * torch.trace(R.T @ L @ R)
            
            print(f"Epoch: {epoch:04d}, train/loss={loss:.5f}")
            
            early_stop(loss)
            if early_stop.isEarlyStopping():
                print(f"Early stopping in round {epoch}")
                break
        
        self.R = R
    
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
    
    
        