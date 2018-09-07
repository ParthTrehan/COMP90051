import math
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from networkx import DiGraph

G = DiGraph()
with open("train.txt", 'r') as f:
    for line in f.read().split('\n'):
        if line:
            line = line.split()
            src = line[0]
            G.add_edges_from((src, dest) for dest in line[1:])
print("---- Read Finish ----")

model = Word2Vec.load("node2vec.model")
print("---- Word2Vec Model Loaded ----")

ids, scores = [], []
k = 5
with open("test-public.txt", 'r') as f:
    lines = f.read().split('\n')
    nn = G.number_of_nodes()
    for line in lines[1:]:
        if line:
            pid, p1, p2 = line.split()
            ids.append(pid)
            print(f"{pid}: {p1} -> {p2}")
            s = model.wv.similarity(p1, p2)
            if s < 0:
                s = 0
            print(f"Random Walk Similarity: {s}")
            scores.append(s)
print("---- Process Finish ----")

s = pd.Series(scores, ids)
s.name = "Prediction"
s.index.name = "Id"
s.to_csv("output.csv", header=True)
