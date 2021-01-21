import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

dngr = rle.DNGR(G)
dngr.initialize()
dngr.initialize_model(d=5)
dngr.fit()
Z = dngr.embed()
print(Z.shape)
print(Z)

Z = rle.DNGR.fast_embed(G)
print(Z.shape)
print(Z)
