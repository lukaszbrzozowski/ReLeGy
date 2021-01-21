import relegy.embeddings as rle
import relegy.graphs as rlg

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)

gr = rle.GraRep(G)
gr.initialize()
gr.fit()
Z = gr.embed()
print(Z.shape)
print(Z)

Z = rle.GraRep.fast_embed(G)
print(Z.shape)
print(Z)
