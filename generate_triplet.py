import csv
from tqdm import tqdm
import numpy as np

import structural_distance as sd


def generate_pair(l):
    pairs = []
    for i in range(len(l)-1):
        for j in range(len(l[i+1:])):
            pairs.append((l[i],l[i+j+1]))
    return pairs


def generate_triplet(q, i, k = 3, window_size = 3):
    dist_fn = sd.get_dist_fn(k = k)

    #with open("random_walks"+str(i)+".csv", 'r', newline='', encoding='utf8') as rf:
    with open("random_walks.csv", 'r', newline='', encoding='utf8') as rf:
        reader = csv.reader(rf)
        a=0###
        num=i
        for row in tqdm(reader, desc="random_walk"):
            for i in range(len(row)-window_size):
                if len(set(row[i:i+window_size+1])) < 3:
                    continue
                anc = int(row[i])
                nodes = np.array([])
                node_dists = ([])
                for n in set(row[i+1:i+window_size+1]):
                    if anc != int(n):                        
                        d = dist_fn(anc, int(n))
                        nodes = np.append(nodes,str(n))
                        node_dists = np.append(node_dists,d)
                
                sample = '_'.join(nodes[node_dists.argsort()])
                triplet = ','.join([str(anc),sample])
                q.put(triplet)
    q.put(None)
    q.close()
    
def write_triplet(q, multi):
    stop = 0
    with open("triplet.csv", 'w', encoding='utf8') as f:
        while True:
            triplet = q.get()
            if triplet is None:
                stop += 1
                if stop == multi:
                    break
                else:
                    continue
            f.write(triplet + '\n')    

def main():
    k = 2#3
    window_size = 7
    #generate_triplet(k = 3, window_size = 3)
    import multiprocessing
    multi = 1
    gen_procs = []
    q = multiprocessing.Queue()
    for i in range(multi):
        g = multiprocessing.Process(target=generate_triplet, args=(q, i, k, window_size))
        g.start()
        gen_procs.append(g)
        
    w = multiprocessing.Process(target=write_triplet, args=(q,multi))
    w.start()
    
    for p in gen_procs:
        p.join()   
    w.join()     
        
if __name__ == '__main__':
    main()