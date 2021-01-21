import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

n2v = rle.Node2Vec(G)
n2v.initialize()
n2v.initialize_model(d=5)
n2v.fit()
Z = n2v.embed()
print(Z.shape)
print(Z)

Z = rle.Node2Vec.fast_embed(G)
print(Z.shape)
print(Z)
