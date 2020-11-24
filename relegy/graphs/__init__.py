import networkx as nx


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
