import os
import csv
import argparse
import multiprocessing
from tqdm import tqdm
import numpy as np

import structural_distance as sd


def load_network(dirpath):
    with open(os.path.join(dirpath, "nodes.csv"), 'r', newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        mapping = {row[1]: int(row[0]) for row in tqdm(reader, desc="load node")}
        
    with open(os.path.join(dirpath, "edges.csv"), 'r', newline='', encoding='utf8') as f: 
        reader = csv.reader(f)
        edges = [(mapping[row[0]], mapping[row[1]]) for row in tqdm(reader, desc="load edge")]
    return edges, len(mapping) + 1


def load_random_walk(filepath):
    with open(filepath, 'r', newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc="random_walk"):
            yield list(map(int, row))


def generate_triplet(q, i, k, window_size, data_dir):
    edges, num_node = load_network(data_dir)
    dist_fn = sd.get_dist_fn(edges, num_node=num_node, k=k)
    a, num = 0, i
    for row in load_random_walk(os.path.join(data_dir, "random_walks.csv")):
        for i in range(len(row)-window_size):
            window = row[i:i+window_size]
            source, contexts = window[0], window[1:]
            structural_distance = [dist_fn(source, context) for context in contexts]
            #print(structural_distance)
            
            sample = ' '.join(map(lambda x: str(x[1]), sorted(zip(structural_distance, contexts))))
            triplet = ','.join([str(source), sample])

            q.put(triplet)
    q.put(None)
    q.close()
    

def write_triplet(q, multi, data_dir):
    stop = 0
    with open(os.path.join(data_dir, "triplet.csv"), 'w', encoding='utf8') as f:
        while True:
            triplet = q.get()
            if triplet is not None:
                f.write(triplet + '\n')    
            else:
                stop += 1
                if stop == multi:
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    k = 2#3
    window_size = 7
    multi = 1
    gen_procs = []
    q = multiprocessing.Queue()
    for i in range(multi):
        g = multiprocessing.Process(target=generate_triplet, args=(q, i, k, window_size, args.data_dir))
        g.start()
        gen_procs.append(g)
        
    w = multiprocessing.Process(target=write_triplet, args=(q, multi, args.data_dir))
    w.start()
    
    for p in gen_procs:
        p.join()   
    w.join()     
        
if __name__ == '__main__':
    main()