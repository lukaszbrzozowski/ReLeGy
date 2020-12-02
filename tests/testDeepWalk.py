from relegy.embeddings import DeepWalk
from relegy.graphs.examples import examplesDict
import numpy as np


def test_deep_walk_fast_embeds_without_error():
    for graph in examplesDict.values():
        Z = DeepWalk.fast_embed(graph)
        assert isinstance(Z, np.ndarray)


def test_deep_walk_result_has_expected_shape():
    for graph in examplesDict.values():
        d = 4
        Z = DeepWalk.fast_embed(graph, d=d)
        n = len(graph.nodes)
        assert Z.shape == (n, d)


def test_deep_walk_parameter_verification():
    graph = None
    try:
        m = DeepWalk(graph)
        assert False
    except Exception:
        assert True

# def test_deep_walk_consistent_embedding_with_identical_random_state():
#     for graph in examplesDict.values():
#         Z1 = DeepWalk.fast_embed(graph, random_seed=2137)
#         Z2 = DeepWalk.fast_embed(graph, random_seed=2137)
#         print(Z1)
#         print(Z2)
#         assert (Z1 == Z2).all()

# def test_deep_walk_different_embedding_with_different_random_state():
#     for graph in examplesDict.values():
#         model1 = DeepWalk(graph, random_state=2137)
#         Z1 = model1.embed()
#         model2 = DeepWalk(graph, random_state=80085)
#         Z2 = model2.embed()
#         assert not (Z1 == Z2).all()
