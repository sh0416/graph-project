import os
import csv
import json
import pprint
import argparse
from collections import Counter
import networkx as nx
from tqdm import tqdm


def load_json_file(filepath):
    with open(filepath, 'r') as f: 
        for line in tqdm(f, desc="load file"):
            yield json.loads(line)


from collections import defaultdict

def process_json(filepaths, output_dir):
    edges = []
    venue_map = {}
    label_map = {}
    """
    venue = defaultdict(set)
    for filepath in filepaths:
        for row in load_json_file(filepath):
            if "authors" in row:
                venue[row["venue"]].update(row["authors"])
    venue = {k: v for k, v in venue.items() if 500 <= len(v) and len(v) < 1500}
    shared_user = Counter()
    for k1, author1 in venue.items():
        for k2, author2 in venue.items():
            if k1 != k2:
                if (k1, k2) in shared_user:
                    continue
                if (k2, k1) in shared_user:
                    continue
                shared_user[k1, k2] += len(author1 & author2)
    for k, count in shared_user.most_common(50):
        print(k, count, len(venue[k[0]]), len(venue[k[1]]), count/len(venue[k[0]] | venue[k[1]]))
    assert False
    """

    author_degree = Counter()
    paper_lists = defaultdict(list)
    for filepath in filepaths:
        for row in load_json_file(filepath):
            if "authors" not in row:
                continue
            if row["venue"] not in ['principles and practice of constraint programming', 'Constraints - An International Journal']:
                continue

            assert len(set(row["authors"])) == len(row["authors"])
            num_authors = len(row["authors"])
            if num_authors < 3:
                continue
            venue_map[row["id"]] = row["venue"]
            for a in row["authors"]:
                edges.append((row["id"], a))
                author_degree[a] += 1
            paper_lists[row["id"]] = row["authors"]
    for paper, authors in paper_lists.items():
        corresponding = max(authors, key=lambda x: author_degree[x])
        for a in authors:
            if a == corresponding:
                label_map[paper, a] = 0
            else:
                label_map[paper, a] = 1

    authors = set(e[0] for e in edges)
    papers = set(e[1] for e in edges)

    # Create largest connected component
    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
    edges = list(graph.edges())

    for i, (x, y) in enumerate(edges):
        if (x in authors) and (y in papers):
            edges[i] = (y, x)
        elif (x in papers) and (y in authors):
            continue
        else:
            assert False
    # computer science logic
    """
    for edge in edges:
        if edge[3] == 'frontiers of combining systems':
            author = edge[1]

    for edge in edges:
        if edge[1] == author:
            print(edge[3])
            """
    """

    with open("venue.csv", 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(venues.most_common())
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "edges.csv"), 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(((x, y, label_map.get((x, y), label_map[(y, x)])) for x, y in edges))

    papers = sorted(list(set(row[0] for row in edges)))
    authors = sorted(list(set(row[1] for row in edges)))
    with open(os.path.join(output_dir, "nodes.csv"), 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(((i, x, "paper") for i, x in tqdm(enumerate(authors, start=1), desc="save paper")))
        writer.writerows(((i, x, "author") for i, x in tqdm(enumerate(papers, start=1+len(authors)), desc="save author")))

    with open(os.path.join(output_dir, "venue.csv"), 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(venue_map.items())


def main():
    """
    node = paper & author
    edge = (paper,author,label)
    label = 1: 1st author
            2: 2nd author
            3: 3rd author
            0: corresponding author
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--prep_dir", type=str, default="prep")
    args = parser.parse_args()

    file_paths = []
    for i in range(4):
        file_paths.append(os.path.join(args.data_dir, "dblp-ref-"+str(i)+".json"))

    process_json(file_paths, args.prep_dir)
    print('finish') 


if __name__ == '__main__':
    main()
