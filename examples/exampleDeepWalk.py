import engthesis.embeddings as emb
import networkx as nx
import matplotlib.pyplot as plt
G = nx.random_graphs.erdos_renyi_graph(200, 0.1)

dw = emb.DeepWalk(G)
dw.initialize(T=40,
              gamma=1)
dw.initialize_model(d=5)
dw.fit()
Z = dw.embed()
print(Z.shape)
Z = emb.DeepWalk.fast_embed(G)
plt.scatter(Z[:, 0], Z[:, 1])
plt.show()
