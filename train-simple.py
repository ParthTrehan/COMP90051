import json
import numpy as np
from keras import Model
from keras.layers import Dense, Input
from keras.models import load_model
from networkx import DiGraph
from networkx.algorithms.simple_paths import all_simple_paths
from pathlib import Path

# file paths
module = "simple"

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


# restore from history
has_history = False
history_path = Path(f"{module}.json")
if history_path.is_file():
    with history_path.open() as f:
        history = json.load(f)
    hsrc, (curr_epoch, max_epoch) = history["src"], history["epoch"]
    model = load_model(f"{module}-{curr_epoch}.h5")
else:
    hsrc, curr_epoch, max_epoch = None, 1, 3

    x = Input((5,))
    y = Dense(1, activation="sigmoid")(x)
    model = Model(x, y)
    model.compile("sgd", "binary_crossentropy")


# training
while curr_epoch <= max_epoch:
    start = True if hsrc is None else False
    for src in G:
        # bypass trained source
        if not start:
            if src == hsrc:
                start = True
            continue

        x = np.array([(
            *neighbour_features(src, dest),
            katz(src, dest)
        ) for dest in G.successors(src)])
        if x.size == 0:
            continue
        y = np.ones((x.shape[0], 1), np.float64)
        model.fit(x, y, epochs=5)
        with history_path.open('w') as f:
            json.dump(
                {"src": src, "epoch": [curr_epoch, max_epoch]}, f, indent=4
            )
        model.save(f"{module}-{curr_epoch}.h5")
    curr_epoch += 1

# with open("test-public.txt") as f:
#     lines = f.read().split('\n')
#     for line in lines[1:]:
#         if line:
#             pid, p1, p2 = [int(i) for i in line.split()]
#             features = bidirectional_dijkstra(G, p1, p2)[0], *neighbour_features(p1, p2), katz(p1, p2)
#             print(f"{pid}: {features} {model.predict(np.array([features]))}")
