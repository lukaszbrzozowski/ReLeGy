import relegy.embeddings as emb
import networkx as nx

G = nx.erdos_renyi_graph(200, 0.1)

GR = emb.GraRep(G)
GR.initialize()
GR.fit(max_K=3)
Z = GR.embed(K=3, d=6)

GR.fit(max_K=5)
Z = GR.embed(K=4, d=8)

GR = emb.GraRep(G, keep_full_SVD=False)
GR.initialize()
GR.fit(max_K=4, d=5)
Z = GR.embed(K=4, concatenated=True)
print(Z.shape)

Z = emb.GraRep.fast_embed(G, d=3, K=2)
print(Z.shape)