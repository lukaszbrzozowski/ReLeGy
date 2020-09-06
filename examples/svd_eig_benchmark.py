import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import time
l = 300
eig = np.empty(300)
svd = np.empty(l)
eig_err = np.empty(l)
svd_err = np.empty(l)
svd_err2 = np.empty(l)
j = 2
for i in tqdm(range(10, l+10)):
    g = nx.random_graphs.barabasi_albert_graph(i, j)
    L = nx.laplacian_matrix(g)
    start = time.time()
    u, v = np.linalg.eig(L.toarray())
    end = time.time()
    eig[i-10] = end-start
    eig_err[i-10] = np.mean(abs(L - v @ np.diag(u) @ v.T))

    start = time.time()
    u, d, v = np.linalg.svd(L.toarray())
    end = time.time()
    svd[i-10] = end-start
    svd_err[i-10] = np.mean(abs(L - u @ np.diag(d) @ v))
    svd_err2[i-10] = np.mean(abs(u - v.T))
    if not i % 100:
        j += 1

plt.plot(np.arange(10, l+10), eig, label = "eig")
plt.plot(np.arange(10, l+10), svd, label = "svd")
plt.legend()
plt.show()

plt.plot(np.arange(10, l+10), eig_err, label = "eig")
plt.plot(np.arange(10, l+10), svd_err, label = "svd")
plt.legend()
plt.show()

plt.plot(np.arange(10, l+10), svd_err2, label="svd_err")
plt.legend()
plt.show()