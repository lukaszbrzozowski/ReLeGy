import engthesis.embeddings as emb
import numpy as np
import networkx as nx

G = nx.erdos_renyi_graph(200, 0.1)
Z = emb.HARP.fast_embed(G)
print(Z.shape)

