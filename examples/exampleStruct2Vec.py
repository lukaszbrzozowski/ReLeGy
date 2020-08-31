import numpy as np
import networkx as nx
from engthesis.embeddings.node.struc2vec import Struct2Vec
import matplotlib.pyplot as plt

bg = nx.barbell_graph(80, 30)
nx.draw(bg, with_labels=True)
plt.show()
s2v = Struct2Vec(bg, d=2, gamma=1, T=20)

Z = s2v.embed()
plt.scatter(Z[:, 0], Z[:, 1])
plt.show()
