import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from engthesis.embeddings.node.node2vec import Node2Vec

np.random.seed(123)
G = nx.random_graphs.erdos_renyi_graph(10, 0.4)
nx.draw(G, with_labels=True)
plt.show()

n2v = Node2Vec(G, d=2, p=10, q=0.01, gamma=2, T=4)
Z = n2v.embed(iter_num=1000, negative=0)
plt.scatter(Z[:, 0], Z[:, 1])
plt.show()
