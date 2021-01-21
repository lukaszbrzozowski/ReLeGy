import relegy.embeddings as rle
import relegy.graphs as rlg
import numpy as np


def factorize(l):
    i = 0
    factorize_dict = {}
    result = []
    for e in l:
        if e in factorize_dict:
            result.append(factorize_dict[e])
        else:
            factorize_dict[e] = i
            result.append(i)
            i += 1
    return result


graph, labels = rlg.get_karate_graph()
Y = np.array([[i for i, label in labels], factorize([label for i, label in labels])]).T

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.001
learning_rate = 0.001
state_dim = 3
max_it = 50
num_epoch = 1000

gnn = rle.GNN(graph=graph)
gnn.initialize(Y)
gnn.initialize_model(embed_dim=state_dim, threshold=threshold, learning_rate=learning_rate, max_it=max_it,
                     mask_flag=False)
gnn.fit(num_epoch=num_epoch)
Z = gnn.embed()
print(Z.shape)
print(Z)

# initialize GNN

Z = rle.GNN.fast_embed(graph, Y,
                       embed_dim=state_dim,
                       num_epoch=num_epoch,
                       threshold=threshold,
                       learning_rate=learning_rate,
                       max_it=max_it,
                       mask_flag=False)

print(Z.shape)
print(Z)
