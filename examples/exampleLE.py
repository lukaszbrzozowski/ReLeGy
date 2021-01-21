import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

le = rle.LaplacianEigenmaps(G)
le.initialize()
le.fit()
Z = le.embed()
print(Z.shape)
print(Z)

Z = rle.LaplacianEigenmaps.fast_embed(G)
print(Z.shape)
print(Z)
