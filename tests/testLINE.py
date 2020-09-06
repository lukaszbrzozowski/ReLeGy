from engthesis.embeddings.node.line import LINE
from engthesis.graphs.examples import examplesDict

def test_line_embeds_without_error():
    for graph in examplesDict.values():
        model = LINE(graph)
        Z = model.embed()
        assert Z is not None
