import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.geznerate_graph("erdos_renyi", n=200, p=0.1)

harp = rle.HARP(G)
harp.initialize()
harp.initialize_model(d=5)
harp.fit()
Z = harp.embed()
print(Z.shape)
print(Z)

Z = rle.HARP.fast_embed(G)
print(Z.shape)
print(Z)
