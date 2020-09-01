import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from engthesis.embeddings.node.graphwave import GraphWave
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
bg = nx.barbell_graph(10, 20)
nx.draw(bg)
plt.show()
gw = GraphWave(bg, 8, J=3, interval_stop=2*np.pi, interval_start=0)
Z = gw.embed()
print(Z.shape)
Z = StandardScaler().fit_transform(Z)
pca = PCA(n_components=2, svd_solver="full")
pc = pca.fit_transform(Z)
plt.scatter(pc[:, 0], pc[:, 1])
plt.show()