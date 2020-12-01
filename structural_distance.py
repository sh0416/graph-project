import os
import csv
import json
import argparse
import pickle
from tqdm import tqdm
import numpy as np

import preprocess
from scipy.sparse import csr_matrix


"""
The part below needs discussion.
"""
#######################################################################

def get_index (dic, key):    
    if key in dic.keys():
        index = dic[key]
    else:
        l = len(dic)/2
        dic[key] = l
        dic[l] = key
        index = l
    return index


def create_row(line):
    """
    :param line:
    :type line: str
    :return: {"authors": list(str), "id": str}
    :rtype:
    """
    row = json.loads(line)
    return {"authors": row.get("authors"), "id": row["id"]}


def process_json2 (file_paths):
    dic = {}
    edges = []
    for file_path in file_paths:
        with open(file_path, 'r') as f: 
            data = list(filter(lambda x : x["authors"] is not None, (create_row(line) for line in tqdm(f, desc="load file"))))

        for row in tqdm(data, desc="extract edge information"):
            row["id"] = get_index (dic, row["id"])
            for i in range(len(row["authors"])):
                row["authors"][i] = get_index (dic, row["authors"][i])
            num_authors = len(row["authors"])
            if num_authors == 1:
                edges.append((row["id"], row["authors"][0], 1))
            elif num_authors == 2:
                edges.append((row["id"], row["authors"][0], 1))
                edges.append((row["id"], row["authors"][1], 0))
            elif num_authors == 3:
                edges.append((row["id"], row["authors"][0], 1))
                edges.append((row["id"], row["authors"][1], 2))
                edges.append((row["id"], row["authors"][2], 0))
            else:
                edges.append((row["id"], row["authors"][0], 1))
                edges.append((row["id"], row["authors"][1], 2))
                for a in row["authors"][2:-1]:
                    edges.append((row["id"], a, 3))
                edges.append((row["id"], row["authors"][-1], 0))

    with open("edges.csv", 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(tqdm(edges, desc="save to csv"))
        
    with open('node2index.pickle','wb') as fw:
        pickle.dump(dic, fw)

###################################################################################################################
"""
The part above needs discussion.
"""


def generate_adjacency_matrix(edges, size):
    rows = [size - 1]
    cols = [size - 1]
    data = [0]
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

def load():
    file_path = './edges.csv'
    edges = []
    with open(file_path, 'r') as f: 
        for line in tqdm(f, desc="load edges"):
            [p, a, l] = line.replace('\n','').split(',')
            edges.append((int(float(p)), int(float(a)), int(float(l))))
    # load dict
    with open('node2index.pickle', 'rb') as fr:
        dic = pickle.load(fr)
    return edges, dic

"""
The function get_degree() needs discussion.
"""
#################################################################################################
def get_degree(adjacency, v, memo = {}):
    return node_degree[v]
    
    
    if v in memo.keys():
        degree = memo[v]
    else:
        degree = sum(adjacency[v].toarray()[0])
        memo[v] = degree
    return degree


def get_neighborhood(adjacency, v):
    return np.nonzero(adjacency[v].toarray()[0])[0]


"""
output: dict
output[n]: n hop neighborhood
"""
def get_k_hop_neighborhood(adjacency, v, k):
    k_hop_neighborhood = {0: set([v])}
    all_neighborhood = set([v])
    for i in range(k):
        prev_neighborhood = k_hop_neighborhood[i]
        i_th_neighborhood = set()
        for u in prev_neighborhood:
            i_th_neighborhood = i_th_neighborhood.union(get_neighborhood(adjacency, u).tolist())
        k_hop_neighborhood[i+1] = i_th_neighborhood - all_neighborhood 
        all_neighborhood = all_neighborhood.union(k_hop_neighborhood[i+1])
        if len(k_hop_neighborhood[i+1]) == 0:
            for j in range(i,k):
                k_hop_neighborhood[j+1] = set()
            return k_hop_neighborhood
    return k_hop_neighborhood


def DTW(X,Y,
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


def get_ordered_degree_seq(adjacency, node_degree, neighbors, memo = {}):
    seq = []
    for i in neighbors:
        seq.append(node_degree[i])
        #seq.append(get_degree(adjacency, i, memo = memo))
    seq.sort()
    return np.array(seq)


def get_structural_distance(adjacency, node_degree, u, v, k, k_hop_u = None, k_hop_v = None):
    if k_hop_u == None:
        k_hop_u = get_k_hop_neighborhood(adjacency, u, k)
    if k_hop_v == None:
        k_hop_v = get_k_hop_neighborhood(adjacency, v, k)
    
    memo = {}
    seq_u = get_ordered_degree_seq(adjacency, node_degree, k_hop_u[k], memo = memo)
    seq_v = get_ordered_degree_seq(adjacency, node_degree, k_hop_v[k], memo = memo)
    
    dist = DTW(seq_u, seq_v)
    
    if k == 0:
        return dist
    return dist + get_structural_distance(adjacency, node_degree, u, v, k-1, k_hop_u, k_hop_v)   


def main():
    file_paths=[]
    for i in range(4):
        file_paths.append("./dblp-ref/dblp-ref-"+str(i)+".json")
    #preprocess.process_json(file_paths)
    process_json2(file_paths)
    ######################################################
    """
    The code above needs discussion.
    """
    u=1
    v=2
    k=3
    
    edges, dic = load()
    num_node = int(len(dic)/2)
    
    adjacency = generate_adjacency_matrix(edges, num_node)
    node_degree = get_node_degree (edges)
    
    dist = get_structural_distance(adjacency, node_degree, u, v, k)
    print(dist)

if __name__ == '__main__':
    main()