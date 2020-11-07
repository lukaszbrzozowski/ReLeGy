from relegy.embeddings import GraphFactorization
from relegy.graphs.examples import examplesDict
import numpy as np

def test_graph_factorization_fast_embeds_without_error():
    for graph in examplesDict.values():
        Z = GraphFactorization.fast_embed(graph)
        assert isinstance(Z, np.ndarray)

def test_graph_factorization_result_has_expected_shape():
    for graph in examplesDict.values():
        d = 4
        Z = GraphFactorization.fast_embed(graph, d=d)
        n = len(graph.nodes)
        assert Z.shape == (n, d)

# def test_graph_factorization_consistent_embedding_with_identical_random_state():
#     for graph in examplesDict.values():
#         model1 = GraphFactorization(graph, random_state=2137)
#         Z1 = model1.embed()
#         model2 = GraphFactorization(graph, random_state=2137)
#         Z2 = model2.embed()
#         assert (Z1 == Z2).all()
#
# def test_graph_factorization_different_embedding_with_different_random_state():
#     for graph in examplesDict.values():
#         model1 = GraphFactorization(graph, random_state=2137)
#         Z1 = model1.embed()
#         model2 = GraphFactorization(graph, random_state=80085)
#         Z2 = model2.embed()
#         assert not (Z1 == Z2).all()