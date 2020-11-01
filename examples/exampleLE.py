import engthesis.embeddings as emb
import networkx as nx
from time import time

G = nx.erdos_renyi_graph(200, 0.1)

start_time = time()
Z = emb.LaplacianEigenmaps.fast_embed(G)
finish_time = time()
print(finish_time-start_time)


start_time = time()
LE = emb.LaplacianEigenmaps(G)
LE.initialize(d=2)
#LE.fit(num_iter=200)
Z = LE.embed()
finish_time = time()
print(finish_time-start_time)
