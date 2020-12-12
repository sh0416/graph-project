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
#from fastdtw import fastdtw
from dtw import accelerated_dtw
###########################################################################################


def generate_adjacency_matrix(edges, size):
    adj = np.zeros((size, size))
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


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
        self.sorted_degree_matrix_list = []
        adj_multi = np.eye(adj.shape[0])
        visit = np.eye(adj.shape[0])
        self.sorted_degree_matrix_list.append(csr_matrix(deg[:, np.newaxis]))
        for i in range(k):
            adj_multi = adj_multi.dot(adj)
            # Order degree
            x = np.where((adj_multi != 0) & (visit == 0),
                         np.ones_like(adj_multi),
                         np.zeros_like(adj_multi))
            visit = x + visit
            degree_matrix = np.where(x != 0, deg[np.newaxis, :], np.zeros_like(x))
            sorted_degree_matrix = np.sort(degree_matrix, axis=1)[:, ::-1]
            sorted_degree_matrix = csr_matrix(sorted_degree_matrix)
            self.sorted_degree_matrix_list.append(sorted_degree_matrix)
        
        self.dist_fn = SoftDTW.SoftDTW().eval()
        self.dist_dic = {}
        self.k = k
        
    def __call__(self, u, v, k=None):
        if (u, v) in self.dist_dic:
            return self.dist_dic[u, v]
        
        k = self.k if k is None else k
        seq_u = self.sorted_degree_matrix_list[k][u].data
        seq_v = self.sorted_degree_matrix_list[k][v].data
        result = accelerated_dtw(seq_u, seq_v, 'euclidean')
        dist = result[0]
        assert dist >= 0, 'dist: %.4f, %s_%s' % (dist, str(seq_u), str(seq_v))
        if k == 0:
            return dist
        elif k == self.k:
            r = dist + self.__call__(u, v, k-1)
            self.dist_dic[u, v] = r
            self.dist_dic[v, u] = r
            return r
        else:
            return dist + self.__call__(u, v, k-1)


def get_dist_fn(edges, k, num_node):
    adj = generate_adjacency_matrix(edges, num_node)
    node_degree = adj.sum(axis=1)
    return StructuralDistance(adj, node_degree, k)
    
    
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
