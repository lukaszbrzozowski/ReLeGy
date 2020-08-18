from engthesis.embeddings.node.harp import HARP
import numpy as np
import networkx as nx

G = nx.random_graphs.barabasi_albert_graph(1000, 2)
harp = HARP(G, T=40, threshold=100)
harp.embed()

