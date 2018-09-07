import numpy as np
import pickle
from networkx import DiGraph
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from operator import itemgetter
from pathlib import Path
from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor

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


print("---- History ----")
hsrc, hdest = None, None
path = Path("SGD.str")
if path.is_file():
    with path.open() as f:
        (hsrc, hdest), fv = eval(f.read())
    print(f"{hsrc}, {hdest}")
    print(fv)
else:
    print("No History")

print("---- Model ----")
mpath = Path("model.pkl")
if mpath.is_file():
    sgd = joblib.load("model.pkl")
    print("Model Loaded")
else:
    sgd = sgd = SGDRegressor(penalty="l1")
    print("Model Created")

print("---- Train ----")
fv, lv = [], []
jump = True
for src, dest in G.edges:
    if hsrc is None and hdest is None:
        jump = False

    if jump:
        if src == hsrc and dest == hdest:
            jump = False
        continue

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
    lv.append(1)

    if len(fv) == 40:
        sgd.partial_fit(np.array(fv), np.array(lv))
        joblib.dump(sgd, "model.pkl")
        fv, lv = [], []

    with path.open('w') as f:
        f.write(str(((src, dest), fv)))
