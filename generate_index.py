import csv
from tqdm import tqdm

def main():
    with open("edges.csv", 'r', newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        data = [row for row in tqdm(reader, desc="load file")]

    authors = sorted(list(set(row[0] for row in data)))
    papers = sorted(list(set(row[1] for row in data)))
    with open("nodes.csv", 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerows(((i, x, "paper") for i, x in tqdm(enumerate(authors, start=1), desc="save paper")))
        writer.writerows(((i, x, "author") for i, x in tqdm(enumerate(papers, start=1+len(authors)), desc="save author")))


if __name__ == "__main__":
    main()
