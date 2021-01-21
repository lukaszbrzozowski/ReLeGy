import relegy.embeddings as rle
import relegy.graphs as rlg
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import relegy.__helpers.gnn_utils as utils


G, labels = rlg.get_karate_graph()
labels = np.array(labels)[:, 1]
labels = (labels == 'Mr. Hi').astype(int)
gcn = rle.GCN(G)
gcn.initialize(Y=labels)
gcn.initialize_model()
gcn.fit()
Z = gcn.embed().numpy()
print(Z.shape)
print(Z)

rle.GCN.fast_embed(G, labels)

print(Z.shape)
print(Z)
