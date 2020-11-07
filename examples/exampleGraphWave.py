import relegy.embeddings as emb
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

bg = nx.barbell_graph(20, 4)
color_map = []
plt.show()
for node in bg:
    if node < 19 or node > 24:
        color_map.append("blue")
    elif node == 19 or node == 24:
        color_map.append("orange")
    elif node == 20 or node == 23:
        color_map.append("green")
    else:
        color_map.append("yellow")
nx.draw(bg, with_labels=True, node_color=color_map)
plt.show()

nx.draw(bg, with_labels=True, node_color=color_map)
plt.show()
gw = emb.GraphWave(bg)
gw.initialize(J=3)
gw.fit(interval_start=0,
       interval_stop=2*np.pi,
       d=20)

Z = gw.embed()
print(Z.shape)
Z = StandardScaler().fit_transform(Z)
pca = PCA(n_components=2, svd_solver="full")
pc = pca.fit_transform(Z)
plt.scatter(pc[:, 0], pc[:, 1], c=color_map)
plt.show()