from relegy.embeddings.node import graphfactorization, laplacianembeddings, hope
from relegy.metrics.metrics import *
import networkx as nx
import matplotlib.pyplot as plt

G = nx.random_graphs.barabasi_albert_graph(1000, 200)
A = nx.to_numpy_array(G)
print("Graph generated")

# le = laplacianembeddings.LaplacianEmbeddings(G, d=5)
# Z_le = le.embed(maxiter=10)
# X_le = Z_le @ Z_le.T
# print("LE trained")

gf = graphfactorization.GraphFactorization(G, d=20)
Z_gf = gf.embed()
X_gf = Z_gf @ Z_gf.T

print("GF trained")

hp = hope.HOPE(G, d=20)
hp.embed()
hp_dict = hp.getMatrixDict()
Us, Ut = hp_dict["Us"], hp_dict["Ut"]

X_hp = Us @ Ut.T

print("HP trained")

# print("Evaluating LE")
# rmse_le = rmse(A, X_le)
# nrmse_le = nrmse(A, X_le)
# prec_k_le = precision_at_k(A, X_le)
# map_le = mean_average_precision(A, X_le)

print("Evaluating GF")
rmse_gf = rmse(A, X_gf)
nrmse_gf = nrmse(A, X_gf)
prec_k_gf = precision_at_k(A, X_gf)
map_gf = mean_average_precision(A, X_gf)

print("Evaluating HOPE")
rmse_hp = rmse(A, X_hp)
nrmse_hp = nrmse(A, X_hp)
prec_k_hp = precision_at_k(A, X_hp)
map_hp = mean_average_precision(A, X_hp)

# plt.plot(np.arange(1, prec_k_le.shape[0]+1), prec_k_le, label="LE")
plt.plot(np.arange(1, prec_k_gf.shape[0]+1), prec_k_gf, label="GF")
plt.plot(np.arange(1, prec_k_hp.shape[0]+1), prec_k_hp, label="HOPE")
plt.legend()
plt.show()

names = ["RMSE", "NRMSE", "MAP"]
plt.bar(names, [rmse_gf, nrmse_gf, map_gf], label="GF")
# plt.bar(names, [rmse_le, nrmse_le, map_le], label="LE")
plt.bar(names, [rmse_hp, nrmse_hp, map_hp], label="HOPE")
plt.legend()
plt.show()

# plt.hist(all_average_precision(A, X_le), label="LE")
# plt.show()

plt.hist(all_average_precision(A, X_gf), label="GF")
plt.show()

plt.hist(all_average_precision(A, X_hp), label="HP")
plt.show()

print(map_gf)
print(map_hp)