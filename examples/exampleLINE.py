import networkx as nx

from engthesis.embeddings.node.line import LINE

G = nx.random_graphs.erdos_renyi_graph(30, 0.2)
L = LINE(G, epochs=1000, alpha1=5e-1, alpha2=1e-3, lmbd1 = 1e-2, lmbd2=1e-3)
Z = L.embed()
