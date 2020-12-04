import os
import csv
import json
import argparse
import pickle
from tqdm import tqdm
import numpy as np

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import preprocess
from scipy.sparse import csr_matrix

###########################################################################################
import SoftDTW
import torch
###########################################################################################


def generate_adjacency_matrix(edges, size):
    rows = [size - 1]
    cols = [size - 1]
    data = [0]
    rows = []
    cols = []
    data = []
    for p, a, _ in tqdm(edges, desc="generate adjacency matrix"):
        rows.append(p)
        rows.append(a)
        
        cols.append(a)
        cols.append(p)
        
        data.append(1)
        data.append(1)    
    adjacency = csr_matrix((data, (rows, cols)))
    return adjacency


def get_node_degree (edges): 
    dic = {}
    for p, a, _ in tqdm(edges, desc="compute node degree"):
        if p in dic.keys():
            dic[p] += 1
        else:
            dic[p]= 1
        if a in dic.keys():
            dic[a] += 1
        else:
            dic[a]= 1
    return dic


class StructuralDistance:
    def __init__(self, adj, deg, k):
        self.adjacency = adj
        self.node_degree = deg
        self.k = k
        #####################################################################
        self.dist_fn = SoftDTW.SoftDTW().eval()
        self.dist_dic = {}
        ##############################################################
        self.a=self.adjacency.dot(self.adjacency)
        
        #self.construct_dist_dic()
    
    
    def get_neighborhood(self, v, adjacency = None):
        if adjacency == None:
            adjacency = self.adjacency
        return np.nonzero(adjacency[v].toarray()[0])[0]
    
    
    """
    output: dict
    output[n]: n hop neighborhood
    """
    def get_k_hop_neighborhood(self, v):
        k_hop_neighborhood = {0: set([v])}
        k_hop_neighborhood[1] = set(self.get_neighborhood(v).tolist())
        all_neighborhood = set([v]).union(set(self.get_neighborhood(v).tolist()))
        k_hop_neighborhood[2] = set(self.get_neighborhood(v,self.a).tolist()) - all_neighborhood
        return k_hop_neighborhood
        ####################
        all_neighborhood = set([v])
        for i in range(self.k):
            prev_neighborhood = k_hop_neighborhood[i]
            i_th_neighborhood = set()
            for u in prev_neighborhood:
                i_th_neighborhood = i_th_neighborhood.union(self.get_neighborhood(u).tolist())
            k_hop_neighborhood[i+1] = i_th_neighborhood - all_neighborhood 
            all_neighborhood = all_neighborhood.union(k_hop_neighborhood[i+1])
            if len(k_hop_neighborhood[i+1]) == 0:
                for j in range(i,self.k):
                    k_hop_neighborhood[j+1] = set()
                return k_hop_neighborhood  
        return k_hop_neighborhood


    def DTW(self,
            X,Y,
            f_dist=lambda x,y:np.sqrt(np.sum((x-y)*(x-y)))
           ):
        #type check
        if (type(X)!=np.ndarray or
            type(Y)!=np.ndarray
           ):
            print('func_DTW: input type error')
            return -1
        #initialize cost matrix
        l_X = len(X)
        l_Y = len(Y)
        cost = float('inf')*np.ones((l_X,l_Y))
        #compute
        cost[0,0]=f_dist(X[0],Y[0])
        for i in range(1,l_X):
            cost[i,0]=cost[i-1,0]+f_dist(X[i],Y[0])
        for j in range(1,l_Y):
            cost[0,j]=cost[0,j-1]+f_dist(X[0],Y[j])
        for i in range(1,l_X):
            for j in range(1,l_Y):
                candidate = cost[i-1,j-1],cost[i,j-1],cost[i-1,j]
                cost[i,j]=min(candidate)+f_dist(X[i],Y[j])
        return cost[-1,-1]


    def get_ordered_degree_seq(self, neighbors):
        seq = []
        for i in neighbors:
            seq.append(self.node_degree[i])
        seq.sort()
        return np.array(seq)    
    
        
    def __call__(self, u, v, k = None, k_hop_u = None, k_hop_v = None):
        update = 0
        if k == None:
            try:
                return self.dist_dic[(u,v)]
            except:
                update = 1
                k = self.k
        if k_hop_u == None:
            k_hop_u = self.get_k_hop_neighborhood(u)
        if k_hop_v == None:
            k_hop_v = self.get_k_hop_neighborhood(v)

        seq_u = self.get_ordered_degree_seq(k_hop_u[k])
        seq_v = self.get_ordered_degree_seq(k_hop_v[k])

        dist = self.dist_fn(torch.Tensor(seq_u).view(1,-1,1),torch.Tensor(seq_v).view(1,-1,1))
        #dist = self.DTW(seq_u, seq_v)
        #dist, _ = fastdtw(np.array(seq_u), np.array(seq_v), dist=euclidean)

        if k == 0:
            return dist
        if update == 1:
            d = dist + self.__call__(u, v, k-1, k_hop_u, k_hop_v)
            self.dist_dic[(u,v)] = d
            self.dist_dic[(v,u)] = d
            return d
        return dist + self.__call__(u, v, k-1, k_hop_u, k_hop_v)   

def get_dist_fn(edges, k = 3, num_node = 4845550):
    adjacency = generate_adjacency_matrix(edges, num_node)
    node_degree = get_node_degree (edges)
    return StructuralDistance(adjacency, node_degree, k)
    
    
def main():
    u=1
    v=2
    k=3
    num_node = 4845550
    
    dist_fn = get_dist_fn(k = 3, num_node = 4845550)
    #dist_fn = StructuralDistance(adjacency, node_degree, k)
    dist = dist_fn(u, v)
    print(dist)

if __name__ == '__main__':
    main()
