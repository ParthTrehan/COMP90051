import math
import numpy as np
import numpy.random as npr
import pandas as pd
from gensim.models import Word2Vec
from networkx import DiGraph


###############################################################################
# https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    K = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand() * K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]
###############################################################################


###############################################################################
# import the graph
G = DiGraph()
with open("train.txt", 'r') as f:
    for line in f.read().split('\n'):
        if line:
            line = line.split()
            src = line[0]
            G.add_edges_from((src, dest) for dest in line[1:])

print("---- Finish Graph Reading ----")
###############################################################################


###############################################################################
# https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf
# https://github.com/aditya-grover/node2vec/blob/master/src/node2vec.py
# initialise node alias sampling
p = q = 1
alias_nodes = {}
for n in G:
    probs = np.ones(G.out_degree(n))
    probs /= probs.sum()
    alias_nodes[n] = alias_setup(probs)
alias_edges = {}
print("---- Finish Nodes Transition Preprocess ----")


def node2vecWalk(G, src, length):
    walk = [src]
    while len(walk) < length:
        curr = walk[-1]
        cn = list(G.successors(curr))
        if G.out_degree(curr) > 0:
            if len(walk) == 1:
                s = alias_nodes[curr]
            else:
                prev = walk[-2]
                co = (prev, curr)
                if co in alias_edges:
                    s = alias_edges[(prev, curr)]
                # initialise edge alias sampling if necessary
                else:
                    def _assign(x):
                        if x == prev:
                            return 1 / p
                        if G.has_edge(x, prev):
                            return 1
                        return 1 / q

                    probs = np.fromiter(map(_assign, cn), np.float64)
                    probs /= probs.sum()
                    s = alias_edges[co] = alias_setup(probs)
            walk.append(cn[alias_draw(*s)])
        else:
            break
    return walk


dimension = 128
epoch = 5
num_walks = 100
walk_length = 10
window_size = 1
workers = 16
model = Word2Vec(
    list(
        node2vecWalk(G, src, walk_length)
        for _ in range(num_walks)
        for src in G
    ),
    size=dimension,
    window=window_size,
    min_count=0,
    workers=workers,
    sg=1,
    hs=1,
    iter=epoch
)
print("---- Finish node2vec ----")

# save the model for future use
model.save("node2vec-3.model")
###############################################################################
