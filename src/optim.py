import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import torch
from utils import *

class ReconstructOptim(): 
    def __init__(self, G, n_iter, attackers, gossip_matrix = LaplacianGossipMatrix, targets_only = False):
        """
        A class to reconstruct the intial values used in a decentralized parallel gd algorithm
        This class depends only on the graph and the attack parameters n_iter and attackers
        It doesn't depend on the actual updates of one particular execution
        G: networkx graph, we require the nodes to be indexed from 0 to n-1
        n_iter: number of gossip iterations n_iter >= 1
        attackers: indices of the attacker nodes
        gossip_matrix: function that returns the gossip matrix of the graph
        """
        self.G = G 
        self.n_iter = n_iter
        self.attackers = attackers
        self.n_attackers = len(attackers)
        self.W = gossip_matrix(self.G)
        self.Wt = torch.tensor(self.W, dtype = torch.float64)
        self.build_knowledge_matrix_dec()
        


    def build_knowledge_matrix_dec(self):
        """
        Building a simplified knowledge matrix including only the targets as unknowns
        This matrix encodes the system of equations that the attackers receive during the learning
        We assume that the n_a attackers appear in the beginning of the gossip matrix
        returns :
            knowledge_matrix : A matrix of shape m * n, where m =  self.n_iter*len(neighbors), n = number of targets
        """
      
        W = self.W
        att_matrix = []
        n_targets = len(self.W) - self.n_attackers
        for neighbor in get_non_attackers_neighbors(self.G, self.attackers):
            att_matrix.append(np.eye(1,n_targets,neighbor-self.n_attackers)[0]) # Shifting the index of the neighbor to start from 0

        pW_TT = np.identity(n_targets)

        for _ in range(1, self.n_iter):
            pW_TT = W[self.n_attackers:,self.n_attackers: ] @ pW_TT + np.identity((n_targets))
            for neighbor in get_non_attackers_neighbors(self.G, self.attackers):  
                att_matrix.append(pW_TT[neighbor-self.n_attackers]) # Assuming this neighbor is not an attacker

        self.target_knowledge_matrix = np.array(att_matrix)
        return self.target_knowledge_matrix


    def build_cov_target_only(self, sigma):  # NewName : Build_covariance_matrix
        """
        Function to build the covariance matrix of the system of equations received by the attackers 
        The number of columns corresponds to the number of targets in the system
        See the pseudo code at algorithm 6 in the report
        return :
            cov : a matrix of size m * m, where m = self.n_iter*len(neighbors)
        """
        W = self.W
        W_TT = W[self.n_attackers:, self.n_attackers:]
        neighbors = get_non_attackers_neighbors(self.G, self.attackers) 

        m = self.n_iter*len(neighbors)

        cov = np.zeros((m,m)) 
        # We iteratively fill this matrix line by line in a triangular fashion (as it is a symetric matrix)
        i = 0
        
        while i < m:
            for it1 in range(self.n_iter):
                for neighbor1 in neighbors:
                    j = it1*len(neighbors)
                    for it2 in range(it1, self.n_iter):
                        for neighbor2 in neighbors:
                            s=0
                            for t in range(it1+1):
                                s+=np.linalg.matrix_power(W_TT,it1+it2-2*t)[neighbor1, neighbor2]
                            cov[i,j] = sigma**2 * s
                            cov[j,i] = cov[i,j]
                            j += 1
                    i+=1
        return cov



    def reconstruct_GLS_target_only(self, v, X_A, sigma):
        """
        Function to reconstruct the inital gradients from the values received by the attackers after self.n_iter iterations.
        This method uses GLS estimator
        v (nd.array) : vector containing the values received by the attackers (in the order defined by the gossip)
        sigma : (float) : variance  
        returns :
            x_hat : a vector of shape n * v.shape[1], where n is the number of nodes
        """
        cov = self.build_cov_target_only(sigma)
        n_targets = len(self.W) - self.n_attackers
        neighbors = np.array(get_non_attackers_neighbors(self.G, self.attackers))
        n_neighbors = len(neighbors)
        v = v[self.n_attackers:] # v[:self.n_attackers] are the attacker sent updates which are the same as X_A[:self.n_attackers]
        d = v[0].shape[0]
        W_TA = self.Wt[self.n_attackers:, :self.n_attackers]
        W_TT = self.Wt[self.n_attackers:, self.n_attackers:]
        pW_TT = np.identity(n_targets, dtype = np.float32)
        new_v = []
        B_t = np.zeros((n_targets, d), dtype = np.float32)
        for it in range(self.n_iter):
            X_A_t = X_A[it*self.n_attackers:(it+1)*self.n_attackers]
            pW_TT = W_TT @ pW_TT + np.identity((n_targets), dtype = np.float32)
            theta_T_t = v[it*n_neighbors:(it+1)*n_neighbors]
            new_v.extend(theta_T_t-B_t[neighbors-self.n_attackers])
            B_t = W_TT @ B_t + W_TA @ X_A_t
        v = np.array(new_v)
        try:
            return GLS(self.target_knowledge_matrix, v, cov)
        except Exception as e:
            print(e)
            print("Building the knowledge matrix failed")
            raise
    

    
    def reconstruct_LS_target_only(self, v, X_A):
        """
        Function to reconstruct the inital gradients from the values received by the attackers after self.n_iter iterations.
        This method uses a Least Squares estimator
        v (nd.array) : vector containing the values received by the attackers (in the order defined by the gossip)
        v looks like (X_A^0, \theta_T^{0+), X_A^1, \theta_T^{1+), ..., X_A^T, \theta_T^{T+)}
        where X_A^t are the attacker sent updates at iteration t and \theta_T^{t+)} are the target sent updates at iteration t
        X_A (nd.array) : vector of size n_a*self.n_iter, containing the attacker sent updates at each iteration
        returns :
            x_hat : a vector of shape n_target * v.shape[1], where n_target is the number of target nodes
        """
        # Prepossessing v to adapt to the target only knowledge matrix

        n_targets = len(self.W) - self.n_attackers
        neighbors = np.array(get_non_attackers_neighbors(self.G, self.attackers))
        n_neighbors = len(neighbors)
        v = v[self.n_attackers:] # v[:self.n_attackers] are the attacker sent updates which are the same as X_A[:self.n_attackers]
        d = v[0].shape[0]
        W_TA = self.Wt[self.n_attackers:, :self.n_attackers]
        W_TT = self.Wt[self.n_attackers:, self.n_attackers:]
        pW_TT = np.identity(n_targets, dtype = np.float32)
        new_v = []
        B_t = np.zeros((n_targets, d), dtype = np.float32)
        for it in range(self.n_iter):
            X_A_t = X_A[it*self.n_attackers:(it+1)*self.n_attackers]
            pW_TT = W_TT @ pW_TT + np.identity((n_targets), dtype = np.float32)
            theta_T_t = v[it*n_neighbors:(it+1)*n_neighbors]
            new_v.extend(theta_T_t-B_t[neighbors-self.n_attackers])
            B_t = W_TT @ B_t + W_TA @ X_A_t

        v = np.array(new_v)

        try:
            return np.linalg.lstsq(self.target_knowledge_matrix, v)[0]
        except Exception as e:
            print(e)
            print("Building the knowledge matrix failed")
            raise


        
