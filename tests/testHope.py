from engthesis.embeddings import HOPE
from engthesis.graphs.examples import examplesDict
import numpy as np

def test_hope_fast_embeds_without_error():
    for graph in examplesDict.values():
        Z = HOPE.fast_embed(graph)
        assert isinstance(Z, np.ndarray)

def test_hope_result_has_expected_shape():
    for graph in examplesDict.values():
        d = 4
        Z = HOPE.fast_embed(graph, d=d)
        n = len(graph.nodes)
        assert Z.shape == (n, 2*d)