import relegy.embeddings as rle
import relegy.graphs as rlr
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import relegy.__helpers.gnn_utils as utils


G, labels = rlr.get_karate_graph()
labels = np.array(labels)[:, 1]
labels = (labels == 'Mr. Hi').astype(int)
gcn = rle.GCN(G)
gcn.initialize(Y=labels)
gcn.initialize_model()
gcn.fit()
Z = gcn.embed().numpy()
print(Z.shape)
print(Z)