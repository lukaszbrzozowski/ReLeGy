from engthesis.embeddings.node.laplacian_embeddings import LaplacianEmbeddings
import networkx as nx

gr1 = nx.complete_graph(5)
gr2 = nx.complete_graph(5)
graph = nx.disjoint_union(gr1,gr2)
graph.add_edge(0, 5)

le = LaplacianEmbeddings(graph)
print(le.embed())