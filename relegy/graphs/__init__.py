import networkx as nx
import numpy as np
import copy


def generate_graph(graph_type, **kwargs):
    """
    Function for generating graphs of specified type;

    Available generators: barbell, complete, erdos_renyi

    barbell parameters:
        m1 - number of nodes in a bell
        m2 - number of nodes in a bar

    complete parameters:
        n - number of nodes

    erdos_renyi
        n - number of nodes
        p - probability for edge creation
    """
    graph_generator_dict = {
        "barbell": nx.barbell_graph,
        "complete": nx.complete_graph,
        "erdos_renyi": nx.random_graphs.erdos_renyi_graph
    }

    return graph_generator_dict[graph_type](**kwargs)


def get_karate_graph():
    karate = nx.karate_club_graph()
    labels = [(node_id, karate.nodes[node_id]["club"]) for node_id in karate.nodes]
    return karate, labels


def generate_clusters_graph(n, k, out_density, in_density):
    """
    Generates a random graph with k clusters which sum to n vertices. The clusters have average edge density equal to in_density
    and the remaining edges between clusters have density equal to out_density.
    Returns the graph and labels corresponding to clusters
    @param n: Number of vertices
    @param k: Number of clusters
    @param out_density: density of graph outside the clusters
    @param in_density: density of clusters
    @return: The generated graph and its labels
    """
    partition = np.random.multinomial(n, np.ones(k)/k, size=1)[0]
    labels = np.repeat(np.arange(k), partition)
    G = nx.Graph()
    cur_min = 0
    G2 = copy.deepcopy(G)
    for i in partition:
        ng = nx.complete_graph(range(cur_min, cur_min+i))
        G.add_nodes_from(ng)
        G2.add_nodes_from(ng)
        G2.add_edges_from(np.array(ng.edges))
        num_edges_left = np.floor(in_density*len(ng.edges)).astype(int)
        edges_ixs_left = np.random.choice(len(ng.edges), num_edges_left, replace=False)
        G.add_edges_from(np.array(ng.edges)[edges_ixs_left, :])
        cur_min += i
    G1 = nx.complement(G2)
    arr = np.arange(len(G1.edges))
    new_edges_size = np.floor(out_density*len(arr))
    new_edges = np.random.choice(arr, size=new_edges_size.astype(int))
    G.add_edges_from(np.array(G1.edges)[new_edges,:])
    return G, labels
