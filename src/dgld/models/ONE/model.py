import numpy as np 
from sklearn.decomposition import NMF
import networkx as nx 
import torch.nn as nn
from utils.early_stopping import EarlyStopping
from .one_utils import loss_func

class ONE():
    """
    ONE (Outlier Aware Network Embedding for Attributed Networks)

    Parameters
    ----------
    node_num : int
        The number of the nodes in graph.
    K : int, optional
        The embedding size, by default 8
    """
    def __init__(self,node_num,K=8):
        self.outl1 = np.ones(node_num)/node_num
        self.outl2 = np.ones(node_num)/node_num
        self.outl3 = np.ones(node_num)/node_num
        self.K = K

    def fit(self,graph,num_epoch=5):
        """
        fit model

        Parameters
        ----------
        graph : dgl.DGLGraph
            The graph you want to embedding.
        num_epoch : int, optional
            Number of epoch, you don't need to much epoch, by default 5
        alpha : float, optional
            Balance parameter , by default None
        beta : float, optional
            Balance parameter , by default None
        gamma : float, optional
            Balance parameter , by default None

        Returns
        -------
        np.array
            The embedding for all node 
        """
        outl1 = self.outl1
        outl2 = self.outl2
        outl3 = self.outl3

        eps = 1e-5

        A = graph.adj().to_dense().numpy()
        C = graph.ndata['feat'].numpy()
        # init W 
        K = self.K
        W = np.eye(K)
        # init G,H by MF for A 
        model = NMF(n_components=self.K, init='random', random_state=0,max_iter=10000)
        G = model.fit_transform(A) 
        H = model.components_

        # init U,V by MF for C 
        model = NMF(n_components=self.K, init='random', random_state=0,max_iter=10000)
        U = model.fit_transform(C)
        V = model.components_

        # init outliers
        outl1 = np.ones((A.shape[0]))
        outl2 = np.ones((A.shape[0]))
        outl3 = np.ones((A.shape[0]))
            
        outl1 = outl1/sum(outl1)
        outl2 = outl2/sum(outl2)
        outl3 = outl3/sum(outl3)

        # init alpha, beta, gamma 

        temp1 = A - np.matmul(G,H)
        temp1 = np.multiply(temp1,temp1)
        temp1 = np.multiply( np.log(np.reciprocal(outl1+eps)), np.sum(temp1, axis=1) )
        temp1 = np.sum(temp1)
                            
        temp2 = C - np.matmul(U,V)
        temp2 = np.multiply(temp2,temp2)
        temp2 = np.multiply( np.log(np.reciprocal(outl2+eps)), np.sum(temp2, axis=1) )
        temp2 = np.sum(temp2)    
        
        temp3 = G.T - np.matmul(W, U.T)
        temp3 = np.multiply(temp3,temp3)
        temp3 = np.multiply( np.log(np.reciprocal(outl3+eps)), np.sum(temp3, axis=0).T )
        temp3 = np.sum(temp3)
        
        alpha = 1
        beta = temp1/temp2
        gamma = min(2*beta, temp3)

        early_stop = EarlyStopping(patience=1,check_finite=False)
        for epoch in range(num_epoch):          
                
            # The Updation rule for G[i,k]    
            for i in range(G.shape[0]):
                for k in range(G.shape[1]):
                    Gik_numer = alpha * np.log(np.reciprocal(outl1[i]+eps)) * np.dot(H[k,:], \
                                        (A[i,:] - (np.matmul(G[i], H) - np.multiply(G[i,k], H[k,:]))) ) + \
                                        gamma * np.log(np.reciprocal(outl3[i]+eps)) * np.dot(U[i], W[k,:])
                    Gik_denom = alpha * np.log(np.reciprocal(outl1[i]+eps)) * np.dot(H[k,:], H[k,:]) + \
                                    gamma * np.log(np.reciprocal(outl3[i]+eps))
                    G[i,k] = Gik_numer / Gik_denom
            
            # The updation rule for H[k,j]
            for k in range(H.shape[0]):
                for j in range(H.shape[1]):
                    Hkj_numer = alpha * np.dot( np.multiply(np.log(np.reciprocal(outl1+eps)), G[:,k]), \
                                        ( A[:,j] - (np.matmul(G,H[:,j]) - np.multiply(G[:,k],H[k,j]) ) ) )
                    Hkj_denom = alpha * ( np.dot(np.log(np.reciprocal(outl1+eps)), np.multiply(G[:,k], G[:,k])) )
                    H[k,j] = Hkj_numer / Hkj_denom
            
            # The up[dation rule for U[i,k]      
            for i in range(U.shape[0]):
                for k in range(U.shape[1]):
                    Uik_numer_1 = beta * np.log(np.reciprocal(outl2[i]+eps)) * ( np.dot( V[k,:],  \
                                                        (C[i] - (np.matmul(U[i,:], V) - np.multiply(U[i,k], V[k,:])) ) ))
                    Uik_numer_2 = gamma * np.log(np.reciprocal(outl3[i]+eps)) * np.dot( \
                                                (G[i,:] - (np.matmul(U[i,:], W) - np.multiply(U[i,k], W[:,k]))), W[:,k] )
                    Uik_denom = beta * np.log(np.reciprocal(outl2[i]+eps)) * np.dot(V[k,:], V[k,:] \
                                                ) + gamma * np.log(np.reciprocal(outl3[i]+eps)) * np.dot(W[:,k], W[:,k])
                    U[i,k] = (Uik_numer_1 + Uik_numer_2) / Uik_denom 
            
            # The updation rule for V[k,d]      
            for k in range(V.shape[0]):
                for d in range(V.shape[1]):
                    Vkd_numer = beta * np.dot( np.multiply(np.log(np.reciprocal(outl2+eps)), U[:,k]), ( C[:,d] \
                                                    - (np.matmul(U,V[:,d]) - np.multiply(U[:,k],V[k,d]) ) ) )
                    Vkd_denom = beta * ( np.dot(np.log(np.reciprocal(outl2+eps)), np.multiply(U[:,k], U[:,k])) )               
                    V[k][d] = Vkd_numer / Vkd_denom        
            
            # The Update rule for W[p,q]
            logoi = np.log(np.reciprocal(outl3+eps))
            sqrt_logoi = np.sqrt(logoi)
            sqrt_logoi = np.tile(sqrt_logoi, (K,1))
            assert(sqrt_logoi.shape == G.T.shape)
            
            term1 = np.multiply(sqrt_logoi, G.T)
            term2 = np.multiply(sqrt_logoi, U.T)
            
            svd_matrix = np.matmul(term1, term2.T)
            
            svd_u, svd_sigma, svd_vt = np.linalg.svd(svd_matrix)
            
            W = np.matmul(svd_u, svd_vt) 
            
            # The updation rule for outl
            GH = np.matmul(G, H)
            UV = np.matmul(U,V)
            WUTrans = np.matmul(W, U.T)
            mu = 1
            outl1_numer = alpha * (np.multiply((A - GH),(A - GH))).sum(axis=1)
            outl1_denom =  alpha * pow(np.linalg.norm((A - GH),'fro'),2)
            outl1_numer = outl1_numer * mu
            outl1 = outl1_numer / outl1_denom
            
            
            outl2_numer = beta * (np.multiply((C - UV),(C - UV))).sum(axis=1)
            outl2_denom =  beta * pow(np.linalg.norm((C - UV),'fro'),2)       
            outl2_numer = outl2_numer * mu
            outl2 = outl2_numer / outl2_denom
            
            outl3_numer = gamma * (np.multiply((G.T - WUTrans),(G.T - WUTrans))).sum(axis=0).T
            outl3_denom = gamma * pow(np.linalg.norm((G.T - WUTrans),'fro'),2)
            outl3_numer = outl3_numer * mu
            outl3 = outl3_numer / outl3_denom
            
            loss = loss_func(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma)
            early_stop(loss)
            if early_stop.isEarlyStopping():
                print(f"Early stopping in round {epoch}")
                break
            # print(loss_func(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma))
            # print ('Loop {} ended: \n'.format(epoch))

        embedding = (G + (U @ W.T))/2
        self.outl1 = outl1 
        self.outl2 = outl2 
        self.outl3 = outl3
        return embedding

    def predict(self, graph, alpha=1.0, beta=1.0, gamma=1.0):
        """
        Predict and return anomaly score of each node

        Parameters
        ----------
        graph : dgl.DGLGraph
            The graph
        alpha : float, optional
            Balance parameter, by default 1.0
        beta : float, optional
            Balance parameter, by default 1.0
        gamma : float, optional
            Balance parameter, by default 1.0

        Returns
        -------
        np.array
            Anomaly score of each node
        """
        return alpha * self.outl1 + beta * self.outl2 + gamma * self.outl3
