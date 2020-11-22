import networkx as nx
import relegy.embeddings as emb


G = nx.erdos_renyi_graph(200, 0.1)

line = emb.LINE(G)
line.initialize(d=2)
line.initialize_model(lmbd1=2)
line.fit()
Z = line.embed()

Z = emb.LINE.fast_embed(G, d=-1)

print(Z.shape)