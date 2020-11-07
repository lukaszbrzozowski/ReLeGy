from relegy.embeddings import SDNE
from relegy.graphs.examples import examplesDict
import numpy as np

def test_sdne_fast_embeds_without_error():
    for graph in examplesDict.values():
        Z = SDNE.fast_embed(graph)
        assert isinstance(Z, np.ndarray)

def test_sdne_result_has_expected_shape():
    for graph in examplesDict.values():
        d = 4
        Z = SDNE.fast_embed(graph, d=d)
        n = len(graph.nodes)
        assert Z.shape == (n, d)