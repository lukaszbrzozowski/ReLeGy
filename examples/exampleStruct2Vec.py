import engthesis.embeddings as emb
import numpy as np
import networkx as nx
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
s2v = emb.Struc2Vec(bg)
s2v.initialize(T=5,
               gamma=20)
s2v.initialize_model(d=10)
s2v.fit(num_iter=3000)
Z = s2v.embed()

pca_ = PCA(n_components=5, svd_solver="full")
pca = pca_.fit_transform(StandardScaler().fit_transform(Z[:, 1:]))
plt.scatter(pca[:, 0], pca[:, 1], c=color_map)
for i, txt in enumerate(np.arange(len(bg.nodes)).astype(str)):
    plt.annotate(txt, (pca[i, 0], pca[i, 1]))
plt.show()
