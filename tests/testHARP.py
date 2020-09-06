from engthesis.embeddings.node.harp import HARP
from engthesis.graphs.examples import examplesDict

def test_harp_embeds_without_error():
    for graph in examplesDict.values():
        model = HARP(graph)
        Z = model.embed()
        assert Z is not None
