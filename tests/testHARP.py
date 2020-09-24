from engthesis.embeddings.node.harp import HARP
from engthesis.graphs.examples import examplesDict


def test_harp_embeds_without_error():
    for graph in examplesDict.values():
        model = HARP(graph)
        Z = model.embed()
        assert Z is not None

def test_harp_consistent_embedding_with_identical_random_state():
    for graph in examplesDict.values():
        model1 = HARP(graph, random_state=2137)
        Z1 = model1.embed()
        model2 = HARP(graph, random_state=2137)
        Z2 = model2.embed()
        assert (Z1 == Z2).all()

def test_harp_different_embedding_with_different_random_state():
    for graph in examplesDict.values():
        model1 = HARP(graph, random_state=2137)
        Z1 = model1.embed()
        model2 = HARP(graph, random_state=80085)
        Z2 = model2.embed()
        assert not (Z1 == Z2).all()