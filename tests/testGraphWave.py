from engthesis.embeddings.node.graphwave import GraphWave
from engthesis.graphs.examples import examplesDict

def test_graph_wave_embeds_without_error():
    for graph in examplesDict.values():
        model = GraphWave(graph)
        Z = model.embed()
        assert Z is not None
