import networkx as nx
import engthesis.embeddings as emb

G = nx.erdos_renyi_graph(200, 0.1)

line = emb.LINE(G)
line.initialize()
line.initialize_model()
line.fit()
Z = line.embed()

Z = emb.LINE.fast_embed(G, d=20)
print(Z.shape)