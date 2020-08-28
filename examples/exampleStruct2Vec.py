import numpy as np
import networkx as nx
from engthesis.embeddings.node.struc2vec import Struct2Vec

bg = nx.barbell_graph(10, 1)

s2v = Struct2Vec(bg)

s2v.generate_similarity_matrices()