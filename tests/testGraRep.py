from engthesis.embeddings.node.grarep import GraRep
from engthesis.graphs.examples import examplesDict

def test_grarep_embeds_without_error():
    for graph in examplesDict.values():
        model = GraRep(graph)
        Z = model.embed()
        assert Z is not None
