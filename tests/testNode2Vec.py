from engthesis.embeddings.node.node2vec import Node2Vec
from engthesis.graphs.examples import examplesDict

def test_node2vec_embeds_without_error():
    for graph in examplesDict.values():
        model = Node2Vec(graph)
        Z = model.embed()
        assert Z is not None

def test_node2vec_consistent_embedding_with_identical_random_state():
    for graph in examplesDict.values():
        model1 = Node2Vec(graph, random_state=2137)
        Z1 = model1.embed()
        model2 = Node2Vec(graph, random_state=2137)
        Z2 = model2.embed()
        assert (Z1 == Z2).all()

def test_node2vec_different_embedding_with_different_random_state():
    for graph in examplesDict.values():
        model1 = Node2Vec(graph, random_state=2137)
        Z1 = model1.embed()
        model2 = Node2Vec(graph, random_state=80085)
        Z2 = model2.embed()
        assert not (Z1 == Z2).all()