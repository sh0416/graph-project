import json

def get_index (node2index, key, final_index):    
    if key in node2index.keys():
        index = node2index[key]
    else:
        node2index[key] = final_index
        index = final_index
    return node2index, index, final_index + 1

def process_json (file_path, node2index, edges, final_index):
    with open(file_path, 'r') as f: 
        for line in f: 
            data = json.loads(line)
            #add paper
            paper_id = data['id']
            node2index, paper, final_index = get_index (node2index, data['id'], final_index)
            
            #add authors
            author_names = data['authors']
            final_author = len(author_names) - 1
            for i, a in enumerate(author_names):
                #add author
                node2index, author, final_index = get_index (node2index, a, final_index)

                #add edge
                if i == 0:
                    edges.append([paper, author, 1])
                elif i == final_author:
                    edges.append([paper, author, 0])
                elif i == 1:
                    edges.append([paper, author, 2])
                else:
                    edges.append([paper, author, 3])
            #print(data['references'])
            #break
            # ref????
    return node2index, edges, final_index

def main():
    """
    node = paper & author
    edge = (paper,author,label)
    label = 1: 1st author
            2: 2nd author
            3: 3rd author
            0: corresponding author
    """

    file_paths = []
    for i in range(4):
        file_paths.append("./dblp-ref/dblp-ref-"+str(i)+".json")

    final_index=0
    edges = []
    node2index = {}

    for f in file_paths:
        node2index, edges, final_index = process_json (f, node2index, edges, final_index)

    print('finish') 

if __name__ == '__main__':
    main()