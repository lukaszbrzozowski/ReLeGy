import relegy.embeddings as emb
import networkx as nx
G = nx.random_graphs.erdos_renyi_graph(100, 0.3)

Z = emb.DNGR.fast_embed(G, d=5, n_layers=-2)
print(Z)