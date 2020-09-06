from engthesis.embeddings.node.graphfactorization import GraphFactorization
from engthesis.graphs.examples import examplesDict

def test_graph_factorization_embeds_without_error():
    for graph in examplesDict.values():
        model = GraphFactorization(graph)
        Z = model.embed()
        assert Z is not None
