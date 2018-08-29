import numpy as np
from keras.models import load_model
from networkx import DiGraph
from networkx.algorithms.simple_paths import all_simple_paths


# import the graph
G = DiGraph()
with open("train.txt", 'r') as f:
    for line in f.read().split('\n'):
        line = line.split()
        if line:
            G.add_edges_from((line[0], i) for i in line[1:])


# features extraction
def neighbour_features(p1, p2):
    ns1, ns2 = set(G.successors(p1)), set(G.successors(p2))
    lns1, lns2 = len(ns1), len(ns2)
    cn = ns1.intersection(ns2)
    lcn = len(cn)

    def _helper(x):
        lsx = G.degree(x)
        return 1 / lsx if lsx > 0 else 0

    # (Common, Jaccard, Adamic/Adar, Preferential)
    return lcn, lcn / len(ns1.union(ns2)), sum(_helper(i) for i in cn), \
        lns1 * lns2


def katz(p1, p2, beta=0.5, cutoff=2):
    pls = {}
    for path in all_simple_paths(G, p1, p2, cutoff):
        lp = len(path)
        pls[lp] = pls.get(lp, 0) + 1
    return sum(beta**l * c for l, c in pls.items())


model = load_model("simple-1.h5")
with open("test-public.txt") as f:
    lines = f.read().split('\n')
    for line in lines[1:]:
        if line:
            pid, p1, p2 = line.split()
            features = *neighbour_features(p1, p2), katz(p1, p2)
            print(f"{pid}: {features} {model.predict(np.array([features]))}")
