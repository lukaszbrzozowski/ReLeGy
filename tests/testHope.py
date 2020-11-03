from engthesis.embeddings import HOPE
from engthesis.graphs.examples import examplesDict
import numpy as np

def test_hope_fast_embeds_without_error():
    for graph in examplesDict.values():
        Z = HOPE.fast_embed(graph)
        assert isinstance(Z, np.ndarray)
