import engthesis.embeddings as emb
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

G = nx.erdos_renyi_graph(20, 0.1)

sdne = emb.SDNE(G)
sdne.initialize(alpha=4)
sdne.initialize_model(n_layers=8, d=2)
sdne.fit(num_iter=6, verbose=True)
Z = sdne.embed()
nA = sdne.get_decoded_matrix()
print(nA)

plt.scatter(Z[:, 0], Z[:, 1])
plt.show()

# ss = StandardScaler()
# Zn = ss.fit_transform(Z)
# pca = PCA(n_components=2)
# pca_f = pca.fit_transform(Zn)
# plt.scatter(pca_f[:, 0], pca_f[:, 1])
# plt.show()
#
# print(Z)