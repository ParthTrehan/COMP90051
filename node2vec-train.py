import gc
import math
import numpy as np
from gensim.models import Word2Vec
from keras.layers import Dense, Input
from keras.models import Model
from keras.utils import Sequence
from networkx import DiGraph
from networkx.linalg.laplacianmatrix import directed_laplacian_matrix

###############################################################################
module = "node2vec-1"
dist_trans = {
    (0, 1): 0,
    (0, 2): 1,
    (1, 1): 2,
    (1, 2): 3,
    (2, 2): 4
}

G = DiGraph()
with open("train.txt", 'r') as f:
    for line in f.read().split('\n'):
        if line:
            line = line.split()
            src = line[0]
            G.add_edges_from((src, dest) for dest in line[1:])
print("---- Read Finish ----")

model = Word2Vec.load(f"{module}.model")
print("---- Word2Vec Model Loaded ----")
###############################################################################


###############################################################################
# https://arxiv.org/pdf/1802.09691.pdf
def extract_enclosing_subgraph(src, dest, h=1):
    vs = {src, dest}
    fringe = {src, dest}
    for _ in range(h):
        if not fringe:
            break
        fringe = {i for v in fringe for i in G.successors(v)} - fringe
        vs.update(fringe)
    return G.subgraph(vs).copy()


x = Input((133,))
y = Dense(2)(x)
predictor = Model(x, y)
predictor.compile("sgd", "mean_squared_error")
for src, dest in G.edges:
    gc.collect()
    sg = extract_enclosing_subgraph(src, dest)

    fv = []
    for n in sg:
        if n == src:
            x = 0
        elif sg.has_edge(n, src):
            x = 1
        else:
            x = 2

        if n == dest:
            y = 0
        elif sg.has_edge(n, dest):
            y = 1
        else:
            y = 2

        dist = np.zeros(5, np.float64)
        dist[dist_trans[(y, x) if x > y else (x, y)]] = 1
        fv.append(np.hstack((dist, model.wv.get_vector(n))))

    # remove to pretend not exist
    sg.remove_edge(src, dest)
    Y = np.zeros((sg.number_of_nodes(), 2), np.uint8)
    Y[:, 1] = 1

    predictor.fit(
        np.matmul(directed_laplacian_matrix(sg), np.vstack(fv)), Y, 2, 2
    )
    predictor.save(f"{module}.h5")
print("---- Model Train Finish ----")
###############################################################################
