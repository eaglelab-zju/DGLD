import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

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
            lr=1e-3,
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
        # feat
        X = graph.ndata['feat']
        X = preprocess_features(X).to(device)
        # adj
        graph = dgl.to_bidirected(graph)
        A = graph.adj().to_dense().to(device)
        
        with torch.no_grad():    
            # Build Laplacian matrix L from the adjacency matrix A
            D = torch.diag(torch.sum(A, dim=1)).to(device)
            L = D - A
            # Initialize D_R and D_W to be identity matrix
            Dr = torch.eye(num_nodes).to(device)
            Dw = torch.eye(num_nodes).to(device)
            # Initialize R = (I + βDR + γL)−1X
            I = torch.eye(num_nodes).to(device)
            R = torch.linalg.solve(I + beta * Dr + gamma * L, X).to(device)
            
            early_stop = EarlyStopping(early_stopping_rounds=2, patience=1, delta=1e-3)
            
            for epoch in range(num_epoch):
                # update w
                W  = torch.linalg.solve(X @ X.T + alpha * Dw, X @ X.T - X @ R.T)
                Dw = torch.diag(1. / (2. * torch.norm(W, dim=1) + eps))
                # update r
                R = torch.linalg.solve(I + beta * Dr + gamma * L, X - W.T @ X)
                Dr = torch.diag(1. / (2. * torch.norm(R, dim=1) + eps))

                loss = torch.norm(X - W.T @ X - R) + alpha * torch.sum(torch.norm(W, dim=1)) + beta * torch.sum(torch.norm(R, dim=1)) + gamma * torch.trace(R.T @ L @ R)
                
                if verbose is True:
                    print(f"Epoch: {epoch:04d}, train/loss={loss:.5f}")
                
                early_stop(loss)
                if early_stop.isEarlyStopping():
                    print(f"Early stopping in round {epoch}")
                    break
            
            self.R = R
        
        # bp
        # self.model = Radar_Base(alpha, beta, gamma)
        # self.model.reset_parameters(A, X)        
        # self.model.w = nn.Parameter(W)
        # self.model.r = nn.Parameter(R)
        
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # for epoch in range(num_epoch):
        #     self.model.train()
        #     loss = self.model(graph, X)
        #     # loss = self.model.loss(X)
            
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
            
        #     if verbose is True:    
        #         print(f"Epoch: {epoch:04d}, train/loss={loss.item():.5f}")
        
        # self.R = self.model.r.cpu().detach() 
    
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
        D = torch.diag(torch.sum(A, dim=1))
        self.l = D - A
        # Initialize r, w
        self.r = nn.Parameter(torch.linalg.solve((1 + self.beta) * torch.eye(num_nodes) + self.gamma * self.l, X))
        self.w = nn.Parameter(torch.eye(num_nodes))
        
    def forward(self, g, x):
        loss = torch.norm(x - self.w.T @ x - self.r) + \
                self.alpha * torch.sum(torch.norm(self.w, dim=1)) + \
                self.beta * torch.sum(torch.norm(self.r, dim=1)) + \
                self.gamma * self.comp_trace(g, x)
                
        return loss
        
    def comp_trace(self, g, x):
        with g.local_scope():
            g.ndata['h'] = self.r
            g.update_all(self._msg_func, fn.sum('dt', 'tr'))
            trace = g.ndata['tr']
         
        trace = torch.sum(trace) * 0.5
        return trace
    
    def _msg_func(self, edges):
        return {'dt': torch.square(edges.src['h'] - edges.dst['h'])}
    
    def loss(self, X):
        loss = torch.norm(X - self.w.T @ X - self.r) + \
                self.alpha * torch.sum(torch.norm(self.w, dim=1)) + \
                self.beta * torch.sum(torch.norm(self.r, dim=1)) + \
                self.gamma * torch.trace(self.r.T @ self.l @ self.r)
                
        return loss