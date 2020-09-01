import numpy as np
import networkx as nx
from engthesis.embeddings.node.struc2vec import Struc2Vec
import matplotlib.pyplot as plt
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
s2v = Struc2Vec(bg, d=5, gamma=20, T=5)

F = s2v.generate_similarity_matrices()
Z = s2v.embed()
print(Z[:, 0])
pca_ = PCA(n_components=5, svd_solver="full")
pca = pca_.fit_transform(StandardScaler().fit_transform(Z[:, 1:]))
plt.scatter(pca[:, 0], pca[:, 1], c=color_map)
for i, txt in enumerate(np.arange(len(bg.nodes)).astype(str)):
    plt.annotate(txt, (pca[i, 0], pca[i, 1]))
plt.show()
