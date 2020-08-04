from engthesis.embeddings.node.laplacian_embeddings import LaplacianEmbeddings
from engthesis.embeddings.node.graph_factorization import GraphFactorization
import networkx as nx
import matplotlib.pyplot as plt
gr1 = nx.complete_graph(5)
gr2 = nx.complete_graph(5)
graph = nx.disjoint_union(gr1, gr2)
graph.add_edge(0, 5)

le = LaplacianEmbeddings(graph, d=5)
print(le.embed())

G = nx.barbell_graph(20, 0)

gf = GraphFactorization(G, d=5)
print(gf.embed())