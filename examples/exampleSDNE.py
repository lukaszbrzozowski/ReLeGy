import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

sdne = rle.SDNE(G)
sdne.initialize()
sdne.initialize_model(d=5)
sdne.fit()
Z = sdne.embed()
print(Z.shape)
print(Z)

Z = rle.SDNE.fast_embed(G)
print(Z.shape)
print(Z)
