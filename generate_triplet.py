import csv
from tqdm import tqdm

import structural_distance as sd


def generate_pair(l):
    pairs = []
    for i in range(len(l)-1):
        for j in range(len(l[i+1:])):
            pairs.append((l[i],l[i+j+1]))
    return pairs


def generate_triplet(q, i, k = 3, window_size = 3):
    dist_fn = sd.get_dist_fn(k = k)

    with open("random_walks"+str(i)+".csv", 'r', newline='', encoding='utf8') as rf:
        reader = csv.reader(rf)
        for row in tqdm(reader, desc="random_walk"):
            for i in range(len(row)-window_size):
                if len(set(row[i:i+window_size+1])) < 3:
                    continue
                anc = int(row[i])
                node_dists = []
                for n in set(row[i+1:i+window_size+1]):
                    if anc != int(n):
                        d = dist_fn(anc, int(n))
                        node_dists.append((int(n),d))
                pairs = generate_pair(node_dists)
                for (p1,d1), (p2,d2) in pairs:
                    if anc == p1 or anc == p2 or p1 == p2:
                        continue
                    #d1 = dist_fn(anc, p1)
                    #d2 = dist_fn(anc, p2)
                    if d1 < d2:
                        pos = p1
                        neg = p2
                    else:
                        pos = p2
                        neg = p1
                    triplet = ','.join([str(anc),str(pos),str(neg)])
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
    k = 3
    window_size = 3    
    #generate_triplet(k = 3, window_size = 3)
    import multiprocessing
    multi = 8
    gen_procs = []
    q = multiprocessing.Queue()
    for i in range(multi):
        g = multiprocessing.Process(target=generate_triplet, args=(q, i, 3, 3))
        g.start()
        gen_procs.append(g)
        
    w = multiprocessing.Process(target=write_triplet, args=(q,multi))
    w.start()
    
    for p in gen_procs:
        p.join()   
    w.join()     
        
if __name__ == '__main__':
    main()