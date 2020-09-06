from engthesis.embeddings.node.hope import HOPE
from engthesis.graphs.examples import examplesDict

def test_hope_embeds_without_error():
    for graph in examplesDict.values():
        model = HOPE(graph)
        Z = model.embed()
        assert Z is not None
