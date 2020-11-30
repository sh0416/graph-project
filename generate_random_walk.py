import csv
import random
import multiprocessing
from collections import defaultdict
from tqdm import tqdm


def producer(q):
    with open("nodes.csv", 'r', newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        node_mapping = {row[1]: int(row[0]) for row in tqdm(reader, desc="load node")}

    adjacency_list = defaultdict(list)
    with open("edges.csv", 'r', newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc="load edge"):
            n1, n2 = node_mapping[row[0]], node_mapping[row[1]]
            adjacency_list[n1].append(n2)
            adjacency_list[n2].append(n1)

    for node_id in tqdm(node_mapping.values(), desc="create_random_walk"):
        random_walk = [node_id]
        for _ in range(100):
            next_node = random.choice(adjacency_list[random_walk[-1]])
            random_walk.append(next_node)
        q.put(','.join(list(map(str, random_walk))))
    q.put(None)
    q.close()
    

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


if __name__ == "__main__":
    main()
