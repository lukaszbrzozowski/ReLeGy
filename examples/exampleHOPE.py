import engthesis.embeddings as emb
import networkx as nx

G = nx.erdos_renyi_graph(200, 0.1)

hope = emb.HOPE(G)
hope.initialize("Katz")
hope.fit()
Z = hope.embed(d=20, concatenated=True)
print(Z.shape)

Z = emb.HOPE.fast_embed(G, d=3, proximity="CN", concatenated=False)
print(Z.shape)