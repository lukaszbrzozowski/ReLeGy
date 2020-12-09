from relegy.embeddings import GCN
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

def test_gcn_fast_embeds_without_error():
    graph, labels = rlg.get_karate_graph()
    Y = np.array(factorize([label for i, label in labels]))
    Z = GCN.fast_embed(graph, Y=Y)
    print(Z.shape)
    assert isinstance(Z, np.ndarray)


def test_gcn_result_has_expected_shape():
    graph, labels = rlg.get_karate_graph()
    Y = np.array(factorize([label for i, label in labels]))
    d = 4
    Z = GCN.fast_embed(graph, Y=Y)
    n = len(graph.nodes)
    assert Z.shape == (n, len(set([label for i, label in labels])))


def test_gcn_parameter_verification():
    graph = None
    try:
        m = GCN(graph)
        assert False
    except Exception:
        assert True