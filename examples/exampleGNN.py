from engthesis.embeddings.node.GNN import GNN
import engthesis.helpers.gnn_utils as utils
import tensorflow.compat.v1 as tf
import networkx as nx

tf.disable_v2_behavior()

##### GPU & stuff config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

############# training set ################
# Provide your own functions to generate input data

E, N, labels, mask_train, mask_test = utils.load_karate()
graph = nx.from_edgelist(list(E[:,:2]))

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.001
learning_rate = 0.001
state_dim = 3
tf.reset_default_graph()
max_it = 50
num_epoch = 10000
optimizer = tf.train.AdamOptimizer

# initialize GNN
param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)

tensorboard = False

g = GNN(graph, labels, state_dim, max_it=max_it, optimizer=optimizer, learning_rate=learning_rate, threshold=threshold, graph_based=False, param=param, config=config,
            tensorboard=tensorboard, mask_flag=True)

result = g.embed()
print(result.shape)