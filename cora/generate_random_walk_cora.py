import csv
import random
import multiprocessing
from collections import defaultdict
from tqdm import tqdm


def producer(q):
    adjacency_list = defaultdict(list)
    with open("edges_cora.csv", 'r', newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc="load edge"):
            n1, n2 = int(row[0]), int(row[1])
            adjacency_list[n1].append(n2)

    for node_id in tqdm(range(2708), desc="create_random_walk"):
        random_walk = [node_id]
        for _ in range(100):
            next_node = random.choice(adjacency_list[random_walk[-1]])
            random_walk.append(next_node)
        q.put(','.join(list(map(str, random_walk))))
    q.put(None)
    q.close()
    
"""
def consumer(q):
    stop = 0
    with open("random_walks.csv", 'w', encoding='utf8') as f:
        while True:
            random_walk = q.get()
            if random_walk is None:
                stop += 1
                if stop == 8:
                    break
                else:
                    continue
            f.write(random_walk + '\n')
            
            
def main():
    q = multiprocessing.Queue()
    producer_procs = []
    for i in range(8):
        p = multiprocessing.Process(target=producer, args=(q,))
        p.start()

    consumer_proc = multiprocessing.Process(target=consumer, args=(q,))
    consumer_proc.start()

    for p in producer_procs:
        p.join()
    consumer_proc.join()
"""


def consumer(q,i):
    stop = 0
    with open("random_walks_cora"+str(i)+".csv", 'w', encoding='utf8') as f:
        while True:
            random_walk = q.get()
            if random_walk is None:
                break
            f.write(random_walk + '\n')

def main():
    producer_procs = []
    consumer_procs = []
    for i in range(8):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=producer, args=(q,))
        p.start()
        producer_procs.append(p)
        consumer_proc = multiprocessing.Process(target=consumer, args=(q,i,))
        consumer_proc.start()
        consumer_procs.append(consumer_proc)

    for p in producer_procs:
        p.join()
        
    for p in consumer_procs:
        p.join()


if __name__ == "__main__":
    main()
