import relegy.embeddings as emb
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import relegy.__helpers.gnn_utils as utils


E, N, labels, mask_train, mask_test = utils.load_karate(path='../data/karate-club/')
colors = {0: "yellow", 1: "green", 2: "blue", 3: "purple"}
order = np.argsort(labels[:, 0])
color_ix = labels[order, 1]
print(colors)
colors_ready = [None] * len(color_ix)
for i in range(len(colors_ready)):
    colors_ready[i] = colors[color_ix[i]]

graph = nx.from_edgelist(list(E[: , :2]))
nx.draw(graph, node_color=colors_ready)
plt.show()

print(emb.GCN.fast_embed(graph, Y=labels[order, 1]))
