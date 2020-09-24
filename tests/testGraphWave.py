from engthesis.embeddings.node.graphwave import GraphWave
from engthesis.graphs.examples import examplesDict


def test_graph_wave_embeds_without_error():
    for graph in examplesDict.values():
        model = GraphWave(graph)
        Z = model.embed()
        assert Z is not None

def test_graph_wave_consistent_embedding_with_identical_random_state():
    for graph in examplesDict.values():
        model1 = GraphWave(graph, random_state=2137)
        Z1 = model1.embed()
        model2 = GraphWave(graph, random_state=2137)
        Z2 = model2.embed()
        assert (Z1 == Z2).all()

def test_graph_wave_different_embedding_with_different_random_state():
    for graph in examplesDict.values():
        model1 = GraphWave(graph, random_state=2137)
        Z1 = model1.embed()
        model2 = GraphWave(graph, random_state=80085)
        Z2 = model2.embed()
        assert not (Z1 == Z2).all()