import networkx as nx

examplesDict = {
    "barbell": nx.barbell_graph(50, 0),
    "complete": nx.complete_graph(100),
    "erdos_renyi": nx.random_graphs.erdos_renyi_graph(100, 0.2)
}
