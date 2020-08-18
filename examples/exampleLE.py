from engthesis.embeddings.node.laplacianembeddings import LaplacianEmbeddings
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
diffs = np.empty(10)
for i in tqdm(range(10)):
    start = time.time()
    G = nx.random_graphs.barabasi_albert_graph(200, 2)
    le = LaplacianEmbeddings(G, d=2)
    Z = le.embed(ftol=1e-8)
    end = time.time()
    diffs[i] = end-start

print(np.mean(diffs))
plt.hist(diffs)
plt.show()
