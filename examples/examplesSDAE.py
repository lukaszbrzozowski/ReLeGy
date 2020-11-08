from relegy.__helpers.sdae import SDAE
import networkx as nx

G = nx.random_graphs.erdos_renyi_graph(100, 0.2)
A = nx.to_numpy_array(G)

sdae = SDAE(n_layers=3, n_hid=[70, 30, 2], optimizer="adam", nb_epoch=20, verbose=0)
model, dat, mse = sdae.get_pretrained_sda(A, get_enc_dec_model=True)
model.compile("adam", "mse")
Z = model.predict(A)

