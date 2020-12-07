import os
import csv
import json
import pprint
import argparse
from collections import Counter
from tqdm import tqdm


def load_json_file(filepath):
    with open(filepath, 'r') as f: 
        for line in tqdm(f, desc="load file"):
            yield json.loads(line)


def process_json(filepaths, output_dir):
    edges = []
    venues = Counter()
    for filepath in filepaths:
        for row in load_json_file(filepath):
            if "authors" not in row:
                continue
            if row["venue"] not in ['computer science logic', 'frontiers of combining systems']:
                continue

            venues[row["venue"]] += 1

            num_authors = len(row["authors"])
            if num_authors == 1:
                edges.append((row["id"], row["authors"][0], 1, row["venue"]))
            elif num_authors == 2:
                edges.append((row["id"], row["authors"][0], 1, row["venue"]))
                edges.append((row["id"], row["authors"][1], 0, row["venue"]))
            else:
                for a in row["authors"][:-1]:
                    edges.append((row["id"], a, 1, row["venue"]))
                edges.append((row["id"], row["authors"][-1], 0, row["venue"]))

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
        writer.writerows(tqdm(edges, desc="save to csv"))


    authors = sorted(list(set(row[0] for row in edges)))
    papers = sorted(list(set(row[1] for row in edges)))
    with open(os.path.join(output_dir, "nodes.csv"), 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(((i, x, "paper") for i, x in tqdm(enumerate(authors, start=1), desc="save paper")))
        writer.writerows(((i, x, "author") for i, x in tqdm(enumerate(papers, start=1+len(authors)), desc="save author")))


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
