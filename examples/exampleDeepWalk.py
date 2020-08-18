import numpy as np
import networkx as nx
from engthesis.embeddings.node.deepwalk import DeepWalk
import matplotlib.pyplot as plt

G = nx.random_graphs.erdos_renyi_graph(30, 0.2)

dw = DeepWalk(G, d=2, T=12, gamma=5)
Z = dw.embed(iter_num=10)
print(Z)
nx.draw(G, with_labels=True)
plt.show()
plt.scatter(Z[:, 0], Z[:, 1])
plt.show()

