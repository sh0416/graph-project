import os
import csv
import networkx as nx
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))

G = nx.Graph()
with open(os.path.join("prep", "nodes.csv"), 'r', newline='', encoding='utf8') as f:
    reader = csv.reader(f)
    nodes = {row[1]: row[2] for row in reader}

with open(os.path.join("prep", "edges.csv"), 'r', newline='', encoding='utf8') as f:
    reader = csv.reader(f)
    edges = []
    labels = []
    for row in reader:
        edges.append((row[0], row[1]))
        labels.append('r' if row[2] == '0' else 'b')

conf_mapping = {}
with open(os.path.join("prep", "venue.csv"), 'r', newline='', encoding='utf8') as f:
    reader = csv.reader(f)
    for row in reader:
        conf_mapping[row[0]] = row[1]

G.add_edges_from(edges)

#Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
#G = G.subgraph(Gcc[0])

node_color = []
for n in G.nodes():
    if nodes[n] == 'author':
        node_color.append('k')
    else:
        if conf_mapping[n] == 'principles and practice of constraint programming':
            node_color.append('r')
        else:
            node_color.append('y')

npos = nx.kamada_kawai_layout(G)
nx.draw_networkx_edges(G, npos, edgelist=edges, edge_color=labels)
nx.draw_networkx_nodes(G, npos, node_size=10, node_color=node_color)

plt.tight_layout()
plt.savefig("hi.png")