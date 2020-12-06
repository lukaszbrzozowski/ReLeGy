from relegy.embeddings.LINEnew import LINEnew
import numpy as np
import networkx as nx
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def generate_clusters_graph(n, k, out_density, in_density):
    """
    Generates a random graph with k clusters which sum to n vertices. The clusters have average edge density equal to in_density
    and the remaining edges between clusters have density equal to out_density.
    Returns the graph and labels corresponding to clusters
    """
    partition = np.random.multinomial(n, np.ones(k)/k, size=1)[0]
    labels = np.repeat(np.arange(k), partition)
    G = nx.Graph()
    cur_min = 0
    G2 = copy.deepcopy(G)
    for i in partition:
        ng = nx.complete_graph(range(cur_min, cur_min+i))
        G.add_nodes_from(ng)
        G2.add_nodes_from(ng)
        G2.add_edges_from(np.array(ng.edges))
        num_edges_left = np.floor(in_density*len(ng.edges)).astype(int)
        edges_ixs_left = np.random.choice(len(ng.edges), num_edges_left, replace=False)
        G.add_edges_from(np.array(ng.edges)[edges_ixs_left, :])
        cur_min += i
    G1 = nx.complement(G2)
    arr = np.arange(len(G1.edges))
    new_edges_size = np.floor(out_density*len(arr))
    new_edges = np.random.choice(arr, size=new_edges_size.astype(int), replace=False)
    print(new_edges)
    G.add_edges_from(np.array(G1.edges)[new_edges,:])
    return G, labels


G, labels = generate_clusters_graph(100, 7, 0.03, 1)
ln = LINEnew(G)
ln.initialize(d=40)
ln.initialize_model(lr2=0.01)
ln.fit(num_iter=800)
Z = ln.embed()

Z = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Z))
plt.scatter(Z[:, 0], Z[:, 1], c=labels)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(Z, labels, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
 .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))