from engthesis.embeddings.node.dngr import DNGR
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
G = nx.random_graphs.erdos_renyi_graph(100, 0.3)

dngr = DNGR(G, d=2)
Z = dngr.embed(n_layers=3, n_hid=[80, 30, 2], nb_epoch=300, dropout=0.3)
plt.scatter(Z[:, 0], Z[:, 1])
plt.show()
print(Z.shape)
print(Z)
