import os
import csv
import json
import argparse
from tqdm import tqdm

def get_index (node2index, key, final_index):    
    if key in node2index.keys():
        index = node2index[key]
    else:
        node2index[key] = final_index
        index = final_index
    return node2index, index, final_index + 1


def create_row(line):
    """
    :param line:
    :type line: str
    :return: {"authors": list(str), "id": str}
    :rtype:
    """
    row = json.loads(line)
    return {"authors": row.get("authors"), "id": row["id"]}


def process_json (file_paths):
    edges = []
    for file_path in file_paths:
        with open(file_path, 'r') as f: 
            data = list(filter(lambda x : x["authors"] is not None, (create_row(line) for line in tqdm(f, desc="load file"))))

        for row in tqdm(data, desc="extract edge information"):
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
