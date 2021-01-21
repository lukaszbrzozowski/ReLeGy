import relegy.embeddings as rle
import relegy.__helpers.gnn_utils as utils
import networkx as nx


##### GPU & stuff config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

############# training set ################
# Provide your own functions to generate input data

E, N, labels, mask_train, mask_test = utils.load_karate()
graph = nx.from_edgelist(list(E[:,:2]))
print(labels)

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.001
learning_rate = 0.001
state_dim = 3
max_it = 50
num_epoch = 1000

gnn = rle.GNN(graph=graph)
gnn.initialize(labels)
gnn.initialize_model(embed_dim=state_dim, num_epoch=num_epoch, threshold=threshold, learning_rate=learning_rate, max_it=max_it, mask_flag=False)
gnn.fit()
Z = gnn.embed
print(Z.shape)
print(Z)

# initialize GNN

Z = rle.GNN.fast_embed(graph, labels,
                        embed_dim=state_dim,
                        num_epoch=num_epoch,
                        threshold=threshold,
                        learning_rate=learning_rate,
                        max_it=max_it,
                        mask_flag=False)

print(Z.shape)
print(Z)