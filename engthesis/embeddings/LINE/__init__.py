import networkx as nx
import numpy as np

from engthesis.model import Model


class LINE(Model):

    def __init__(self, graph, d=2, alpha1=1e-4, alpha2=1e-4, epochs=400,
                 batch_size=30, lmbd1=1e-1, lmbd2=1e-2, verbose=False, **kwargs):
        super().__init__(graph.to_directed())
        self.__d = d
        self.__alpha1 = alpha1
        self.__alpha2 = alpha2
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__lmbd1 = lmbd1
        self.__lmbd2 = lmbd2
        self.__verbose = verbose
        self.__A = kwargs["A"] if "A" in kwargs else nx.to_numpy_array(self.get_graph())
        self.__U1 = kwargs["U1"] if "U1" in kwargs else np.random.random((len(graph.nodes), self.__d))
        self.__U2 = kwargs["U2"] if "U2" in kwargs else np.random.random((len(graph.nodes), self.__d))
        self.__Z = None

    def embed(self):
        Frob = lambda U: np.linalg.norm(U)
        graph = self.get_graph()
        n_edges = len(graph.edges)
        p1 = lambda x, y: 1 / (1 + np.exp(-np.dot(x, y)))
        p2 = lambda x, y: np.exp(np.dot(x, y)) / (np.sum(self.__U2 @ y))
        o1 = lambda G: -sum([self.__A[i, j] * np.log(p1(self.__U1[i, :], self.__U1[j, :])) for (i, j) in G.edges])
        o2 = lambda G: -sum([self.__A[i, j] * np.log(p2(self.__U2[i, :], self.__U2[j, :])) for (i, j) in G.edges])
        grad1 = lambda i, j: self.__A[i, j] * (-self.__U1[j, :]) * np.exp(-np.dot(self.__U1[i, :], self.__U1[j, :])) / (
                    1 + np.exp(-np.dot(self.__U1[i, :], self.__U1[j, :]))) ** 2
        grad2 = lambda i, j: self.__A[i, j] * (
                    self.__U2[j, :] * np.exp(np.dot(self.__U2[i, :], self.__U2[j, :])) * np.sum(
                self.__U2 @ self.__U2[j, :]) - np.exp(np.dot(self.__U2[i, :], self.__U2[j, :])) * self.__U2[j, :]) / (
                                 np.sum(self.__U2 @ self.__U2[j, :])) ** 2
        for iter in range(self.__epochs):
            resampled_edges = np.copy(graph.edges)
            np.random.shuffle(resampled_edges)
            mini_batches = [resampled_edges[i * self.__batch_size:min(n_edges, i + 1 * self.__batch_size)] for i in
                            range(n_edges // self.__batch_size)]
            for batch in mini_batches:
                step1 = np.zeros(self.__U1.shape)
                step2 = np.zeros(self.__U2.shape)
                for (i, j) in batch:
                    step1[i, :] += grad1(i, j) + self.__lmbd1 * self.__U1[i, :]
                    step2[i, :] += grad2(i, j) + self.__lmbd2 * self.__U2[i, :]
                self.__U1 -= self.__alpha1 * step1
                self.__U2 -= self.__alpha2 * step2
            if self.__verbose: print(
                f"Epoch: {iter + 1}, 1st-order objective: {o1(graph):.2f}, 1st-order Frob:{Frob(self.__U1):.2f} 2nd-order objective: {o2(graph):.2f}, 2nd-order Frob:{Frob(self.__U2):.2f}",
                end="\r")
        self.__Z = np.c_[self.__U1, self.__U2]
        return self.__Z

    def info(self):
        pass
