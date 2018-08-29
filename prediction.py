from networkx import DiGraph

# import the graph
G = DiGraph()
with open("train.txt", 'r') as f:
    for line in f.read().split('\n'):
        line = line.split()
        if line:
            G.add_edges_from((line[0], i) for i in line[1:])
