import math
import numpy as np
import pandas as pd
import random
from gensim.models import Word2Vec
from networkx import DiGraph
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from sklearn.linear_model import SGDRegressor
from time import time

start = time()
G = DiGraph()
with open("train.txt", 'r') as f:
    for line in f.read().split('\n'):
        if line:
            line = line.split()
            src = line[0]
            G.add_edges_from((src, dest) for dest in line[1:])
print(f"---- Read Finish in {time() - start:.2f}s ----")

start = time()
prs = pagerank(G)
print(f"---- PageRank Finish in {time() - start:.2f}s ----")

start = time()
model = Word2Vec.load("node2vec.model")
model1 = Word2Vec.load("node2vec-1.model")
model2 = Word2Vec.load("node2vec-2.model")
print(f"---- Word2Vec Model Loaded in {time() - start:.2f}s ----")

K = 20000


def generate_features(src, dest):
    nsrc = set(G.successors(src))
    ndest = set(G.predecessors(dest))
    cn = nsrc.intersection(ndest)
    lcn = len(cn)
    return [
        model.wv.similarity(src, dest),
        model1.wv.similarity(src, dest),
        model2.wv.similarity(src, dest),
        sum(1 / math.log(G.out_degree(z) + 1) for z in cn),
        lcn,
        lcn / len(nsrc.union(ndest)) if lcn else 0,
        prs[src],
        prs[dest],
        G.in_degree(src),
        G.in_degree(dest),
        G.out_degree(src),
        G.out_degree(dest),
        G.has_edge(dest, src)
    ]


start = time()
data = [[*generate_features(*edge), 1] for edge in random.sample(G.edges, K)]
nodes = list(G.nodes)


def sample_discon_edge():
    src, dest = random.choice(nodes), random.choice(nodes)
    while G.has_edge(src, dest) or src == dest:
        dest = random.choice(nodes)
    # may generate duplicates
    return src, dest


data.extend([
    [*generate_features(*sample_discon_edge()), 0]
    for _ in range(K)
])
data = np.array(data)
np.random.shuffle(data)
print(f"---- Sampling Finish in {time() - start:.2f}s ----")

start = time()
regressor = SGDRegressor()
regressor.fit(data[:, :-1], data[:, -1])
print(f"---- Train Finish in {time() - start:.2f}s ----")

start = time()
ids, scores = [], []
with open("test-public.txt") as f:
    lines = f.read().split('\n')
    for line in lines[1:]:
        if line:
            pid, src, dest = list(line.split())
            ids.append(pid)
            print(f"{pid}: {src} -> {dest}")
            s = regressor.predict([generate_features(src, dest)])[0]
            if s < 0:
                s = 0
            elif s > 1:
                s = 1
            print(f"Logistic Regressor: {s}")
            scores.append(s)
print(f"---- Process Finish in {time() - start:.2f}s ----")

s = pd.Series(scores, ids)
s.name = "Prediction"
s.index.name = "Id"
s.to_csv("output-smaller.csv", header=True)
