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

model = Word2Vec.load("node2vec-1.model")
print("---- Word2Vec Model Loaded ----")

ids, srcs, dests = [], [], []
with open("test-public.txt", 'r') as f:
    lines = f.read().split('\n')
    nn = G.number_of_nodes()
    for line in lines[1:]:
        if line:
            pid, p1, p2 = line.split()
            ids.append(pid)
            srcs.append(p1)
            dests.append(p2)
print("---- Preload Finish ----")

es = np.zeros((len(ids), 128), np.float64)
for w in G:
    wh = sum(model.wv.get_vector(i) for i in G.successors(w))

    for i, (src, dest) in enumerate(zip(srcs, dests)):
        sim = model.wv.similarity(src, w)
        if sim > 0.5:
            es[i] += 0.5 * sim * wh
        sim = model.wv.similarity(dest, w)
        if sim > 0.5:
            es[i] += 0.5 * sim * wh
scores = [
    np.dot(es[i], model.wv.get_vector(dest)) for i, dest in enumerate(dests)
]
print("---- Process Finish ----")
print(scores)

s = pd.Series(scores, ids)
s /= s.groupby(srcs).transform(max)
s[s < 0] = 0
s[s > 1] = 1
s.fillna(0)
s.name = "Prediction"
s.index.name = "Id"
s.to_csv("output.csv", header=True)
