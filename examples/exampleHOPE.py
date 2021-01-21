import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

hope = rle.HOPE(G)
hope.initialize()
hope.fit()
Z = hope.embed()
print(Z.shape)
print(Z)

Z = rle.HOPE.fast_embed(G)
print(Z.shape)
print(Z)
