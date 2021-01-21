import relegy.embeddings as rle
import relegy.metrics as rlm
import relegy.graphs as rlg
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = rlg.generate_graph("erdos_renyi", n=200, p=0.1)
A = nx.to_numpy_array(G)
print("Graph generated")

Z_gf = rle.GraphFactorization.fast_embed(G, fit_verbose=False)
X_gf = Z_gf @ Z_gf.T

print("GF trained")

Z_hp = rle.HOPE.fast_embed(G)
Us = Z_hp[:, :2]
Ut = Z_hp[:, 2:]
X_hp = Us @ Ut.T
print("HP trained")

print("Evaluating GF")
rmse_gf = rlm.rmse(A, X_gf)
nrmse_gf = rlm.nrmse(A, X_gf)
prec_k_gf = rlm.precision_at_k(A, X_gf)
map_gf = rlm.mean_average_precision(A, X_gf)

print("Evaluating HOPE")
rmse_hp = rlm.rmse(A, X_hp)
nrmse_hp = rlm.nrmse(A, X_hp)
prec_k_hp = rlm.precision_at_k(A, X_hp)
print(A)
print(X_hp)
map_hp = rlm.mean_average_precision(A, X_hp)


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

plt.hist(rlm.all_average_precision(A, X_gf), label="GF")
plt.show()

plt.hist(rlm.all_average_precision(A, X_hp), label="HP")
plt.show()

print(map_gf)
print(map_hp)
