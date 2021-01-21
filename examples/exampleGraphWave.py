import relegy.embeddings as rle
import relegy.graphs as rlg
import matplotlib.pyplot as plt

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

gw = rle.GraphWave(G)
gw.initialize()
gw.fit()
Z = gw.embed()
print(Z.shape)
Z = rle.GraphWave.fast_embed(G)
print(Z.shape)
print(Z)
