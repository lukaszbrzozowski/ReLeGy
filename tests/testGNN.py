from relegy.embeddings.GNN import GNN
import relegy.graphs as rlg
import numpy as np

def factorize(l):
    i = 0
    factorize_dict = {}
    result = []
    for e in l:
        if e in factorize_dict:
            result.append(factorize_dict[e])
        else:
            factorize_dict[e] = i
            result.append(i)
            i += 1
    return result

def test_gnn_fast_embeds_without_error():
    graph, labels = rlg.get_karate_graph()
    Y = np.array([[i for i, label in labels], factorize([label for i, label in labels])]).T
    Z = GNN.fast_embed(graph, idx_labels=Y)
    assert isinstance(Z, np.ndarray)


def test_gnn_result_has_expected_shape():
    graph, labels = rlg.get_karate_graph()
    Y = np.array([[i for i, label in labels], factorize([label for i, label in labels])]).T
    d = 4
    Z = GNN.fast_embed(graph, idx_labels=Y, embed_dim=d)
    n = len(graph.nodes)
    print(Z.shape)
    assert Z.shape == (n, d)


def test_gnn_parameter_verification():
    graph = None
    try:
        m = GNN(graph)
        assert False
    except Exception:
        assert True