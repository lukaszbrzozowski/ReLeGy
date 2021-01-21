import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

dw = rle.DeepWalk(G)
dw.initialize(T=40,
              gamma=1)
dw.initialize_model(d=5)
dw.fit()
Z = dw.embed()
print(Z.shape)
print(Z)

Z = rle.DeepWalk.fast_embed(G, negative=-1)
print(Z.shape)
print(Z)
