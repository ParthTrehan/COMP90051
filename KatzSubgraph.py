import math
import numpy as np
import pandas as pd
from collections import Counter
from networkx import DiGraph
from networkx.algorithms.simple_paths import all_simple_paths

# import the graph
G = DiGraph()
with open("train.txt", 'r') as f:
    for line in f.read().split('\n'):
        if line:
            line = line.split()
            src = line[0]
            G.add_edges_from((src, dest) for dest in line[1:])
print("---- Read Finish ----")


def extract_enclosing_subgraph(src, dest, h=1):
    vs = {src, dest}
    fringe_out, fringe_in = {src}, {dest}
    for _ in range(h):
        if not fringe:
            break
        fringe_out = {
            i for v in fringe_out for i in G.successors(v)
        } - fringe_out
        fringe_in = {
            i for v in fringe_in for i in G.predecessors(v)
        } - fringe_in
        vs.update(fringe_in, fringe_out)
    return G.subgraph(vs).copy()


penalty = 4
beta = 1 / penalty
ids, scores = [], []
with open("test-public.txt") as f:
    lines = f.read().split('\n')
    for line in lines[1:]:
        if line:
            pid, p1, p2 = list(line.split())
            ids.append(pid)
            print(f"{pid}: {p1} -> {p2}")
            s = math.tanh(sum(
                2 * penalty * beta**l * c for l, c in Counter(
                    len(p) for p in all_simple_paths(G, p1, p2, 2)
                ).items()
            ))
            print(f"tanh(Katz) Subgraph: {s}")
            scores.append(s)
print("---- Process Finish ----")

s = pd.Series(scores, ids)
s.name = "Prediction"
s.index.name = "Id"
s.to_csv("output.csv", header=True)
