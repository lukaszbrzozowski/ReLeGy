import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

line = rle.LINE(G)
line.initialize()
line.fit()
Z = line.embed()
print(Z.shape)
print(Z)

Z = rle.LINE.fast_embed(G)
print(Z.shape)
print(Z)
