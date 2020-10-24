import engthesis.embeddings as emb
import numpy as np
import networkx as nx

G = nx.erdos_renyi_graph(200, 0.1)
harp = emb.HARP(G)
harp.initialize(method="DeepWalk",
                T=40,
                gamma=1)
harp.initialize_model(d=2)
harp.fit(num_iter=300)
Z = harp.embed()
print(Z.shape)

