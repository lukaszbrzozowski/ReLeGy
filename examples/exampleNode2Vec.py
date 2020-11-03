import networkx as nx
import engthesis.embeddings as emb

G = nx.erdos_renyi_graph(200, 0.1)

n2v = emb.Node2Vec(G)
n2v.initialize()
n2v.initialize_model(d = 5)
n2v.fit()
Z = n2v.embed()
print(Z.shape)