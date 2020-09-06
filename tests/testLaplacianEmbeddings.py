from engthesis.embeddings.node.laplacianembeddings import LaplacianEmbeddings
from engthesis.graphs.examples import examplesDict

def test_laplacian_embeddings_embeds_without_error(self):
    for graph in examplesDict.values():
        model = LaplacianEmbeddings(graph)
        Z = model.embed()
        assert Z is not None
