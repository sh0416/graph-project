import csv
from tqdm import tqdm


def process_edges ():
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset.data.edge_index
    edges = []
    for i in tqdm(range(dataset.data.edge_index.shape[1]), desc="extract edge information"):
        edges.append((int(data[0,i]), int(data[1,i]), 1))
        edges.append((int(data[1,i]), int(data[0,i]), 0))

    with open("edges_cora.csv", 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(tqdm(edges, desc="save to csv"))    


def main():
    """
    # of node = 2708
    node = paper & author
    edge = (paper,author,label)
    label = 1: cited
            0: citing
    """
    process_edges ()


if __name__ == '__main__':
    main()
