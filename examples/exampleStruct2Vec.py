import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

struc2vec = rle.Struc2Vec(G)
struc2vec.initialize()
struc2vec.initialize_model(d=5)
struc2vec.fit()
Z = struc2vec.embed()
print(Z.shape)
print(Z)

Z = rle.Struc2Vec.fast_embed(G)
print(Z.shape)
print(Z)
