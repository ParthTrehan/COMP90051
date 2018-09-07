import math
import pandas as pd
from networkx import DiGraph

# import the graph
G = DiGraph()
with open("train.txt", 'r') as f:
    for line in f.read().split('\n'):
        if line:
            line = line.split()
            src = line[0]
            G.add_edges_from((src, dest) for dest in line[1:])
print("-- Read Finish --")


def jaccard(p1, p2):
    n1, n2 = set(G.successors(p1)), set(G.successors(p2))
    un = n1.union(n2)
    return len(n1.intersection(n2)) / len(n1.union(n2)) if un else 0


ids, scores = [], []
with open("test-public.txt") as f:
    lines = f.read().split('\n')
    for line in lines[1:]:
        if line:
            pid, p1, p2 = list(line.split())
            ids.append(pid)
            print(f"{pid}: {p1} -> {p2}")
            s = math.tanh(sum(jaccard(n, p2) for n in G.successors(p1)))
            print(f"Jaccard: {s}")
            scores.append(s)
print("-- Process Finish --")

s = pd.Series(scores, ids)
s.name = "Prediction"
s.index.name = "Id"
s.to_csv("output.csv", header=True)
