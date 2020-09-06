from engthesis.embeddings.node.deepwalk import DeepWalk
from engthesis.graphs.examples import examplesDict

def test_deep_walk_embeds_without_error():
    for graph in examplesDict.values():
        model = DeepWalk(graph)
        Z = model.embed()
        assert Z is not None
