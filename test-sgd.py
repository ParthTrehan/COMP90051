import numpy as np
from networkx import DiGraph
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from operator import itemgetter
from sklearn.externals import joblib


G = DiGraph()
with open("train.txt", 'r') as f:
    for line in f.read().split('\n'):
        if line:
            line = line.split()
            src = line[0]
            G.add_edges_from((src, dest) for dest in line[1:])

print("---- Finish Graph Reading ----")

prs = pagerank(G)

print("---- Finish Page Rank ----")


def similarity(p1, p2):
    n1, n2 = set(G.successors(p1)), set(G.successors(p2))
    cn = n1.intersection(n2)
    un = n1.union(n2)
    return p1 in n2, p2 in n1, G.degree(p2), prs[p2], \
        len(cn) / len(un) if un else 0, sum(1 / G.degree(z) for z in cn)


print("---- Test ----")
sgd = joblib.load("model.pkl")
fv = []
with open("test-public.txt") as f:
    lines = f.read().split('\n')
    for line in lines[1:]:
        if line:
            pid, src, dest = line.split()
            f = [G.degree(dest), prs[dest], *similarity(dest, src)[1:]]
            sf = sorted(
                (similarity(dest, s) for s in G.successors(src)),
                key=itemgetter(3, 2, 5)
            )[-3:]
            l = len(sf)
            if l < 3:
                sf = [tuple([0] * 6)] * (3 - l) + sf
            for i in sf:
                f.extend(i)
            fv.append(f)

print(sgd.predict(np.array(fv)))
