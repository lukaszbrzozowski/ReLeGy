from engthesis.embeddings.node.node2vec import Node2Vec
from engthesis.graphs.examples import examplesDict

def test_node2vec_embeds_without_error():
    for graph in examplesDict.values():
        model = Node2Vec(graph)
        Z = model.embed()
        assert Z is not None