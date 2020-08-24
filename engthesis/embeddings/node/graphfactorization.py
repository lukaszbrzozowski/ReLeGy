import networkx as nx
import numpy as np
from engthesis.model.base import Model
from scipy.sparse import csr_matrix

class GraphFactorization(Model):

    def __init__(self, graph, d=2, eps=1e-7, lmbd=0) -> None:
        """

        :rtype: object
        """
        super().__init__(graph)
        self.__A: csr_matrix = nx.to_numpy_array(self.get_graph())
        self.__d: int = d
        self.__eps: float = eps
        self.__lmbd: float = lmbd


    def info(self) -> str:
        return "To be implemented"

    def embed(self) -> np.ndarray:
        G = self.get_graph()
        n = len(G.nodes)
        Z = np.random.rand(n, self.__d)
        t = 1
        error = np.infty
        while self.__eps >= error:
            Z_previous = np.copy(Z)
            edges = list(G.edges)
            np.random.shuffle(edges)
            for i, j in edges:
                eta = 1/np.sqrt(t)
                t += 1
                Z[i, ] += eta*((self.__A[i, j]-np.dot(Z[i, ], Z[j, ])) * Z[j, ] - self.__lmbd * Z[i, ])
                t += 1
                Z[j, ] = Z[j, ] + eta * ((self.__A[i, j] - np.dot(Z[i, ], Z[j, ])) * Z[i, ] - self.__lmbd * Z[j, ])
            error = np.sum((Z-Z_previous)**2)
            print(error, end="\r")
        return Z

