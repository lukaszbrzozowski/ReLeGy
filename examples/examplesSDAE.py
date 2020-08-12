from engthesis.helpers.sdae import SDAE
import networkx as nx

G = nx.random_graphs.erdos_renyi_graph(100, 0.2)
A = nx.to_numpy_array(G)

sdae = SDAE(n_layers=3, n_hid=[70, 30, 2], optimizer="adam", nb_epoch=20)
model, dat, mse = sdae.get_pretrained_sda(A, dir_out=None, write_model=False)
model.compile("adam", "mse")
Z = model.predict(A)
print(Z.shape)
print(Z)


