import os
import csv
import json
import argparse
from collections import Counter
from tqdm import tqdm

def get_index (dic, key):    
    if key in dic.keys():
        index = dic[key]
    else:
        l = len(dic)/2
        dic[key] = l
        dic[l] = key
        index = l
    return index


def load_json_file(filepath):
    with open(filepath, 'r') as f: 
        for line in tqdm(f, desc="load file"):
            yield json.loads(line)


def process_json (filepaths):
    edges = []
    venues = Counter()
    for filepath in filepaths:
        for row in load_json_file(filepath):
            if "authors" not in row:
                continue
            if row["venue"] != 'IEEE Transactions on Pattern Analysis and Machine Intelligence':
                continue
            venues[row["venue"]] += 1

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

    print(venues.most_common())
    os.makedirs("prep", exist_ok=True)
    with open(os.path.join("prep", "edges.csv"), 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(tqdm(edges, desc="save to csv"))

    authors = sorted(list(set(row[0] for row in edges)))
    papers = sorted(list(set(row[1] for row in edges)))
    with open(os.path.join("prep", "nodes.csv"), 'w', newline='', encoding='utf8') as f:
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
    args = parser.parse_args()

    file_paths = []
    for i in range(4):
        file_paths.append(os.path.join(args.data_dir, "dblp-ref-"+str(i)+".json"))

    process_json(file_paths)
    print('finish') 


if __name__ == '__main__':
    main()
