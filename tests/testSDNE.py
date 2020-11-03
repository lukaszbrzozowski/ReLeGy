from engthesis.embeddings import SDNE
from engthesis.graphs.examples import examplesDict
import numpy as np

def test_sdne_fast_embeds_without_error():
    for graph in examplesDict.values():
        Z = SDNE.fast_embed(graph)
        assert isinstance(Z, np.ndarray)
