from engthesis.embeddings.node.struc2vec import Struc2Vec
from engthesis.graphs.examples import examplesDict

def test_struc2vec_embeds_without_error():
    for graph in examplesDict.values():
        model = Struc2Vec(graph)
        Z = model.embed()
        assert Z is not None
