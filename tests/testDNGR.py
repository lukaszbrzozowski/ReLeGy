from engthesis.embeddings.node.dngr import DNGR
from engthesis.graphs.examples import examplesDict

def test_dngr_embeds_without_error():
    for graph in examplesDict.values():
        model = DNGR(graph)
        Z = model.embed()
        assert Z is not None
