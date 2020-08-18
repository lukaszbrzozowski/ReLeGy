from networkx import adjacency_matrix, Graph
import numpy as np
from engthesis.model.base import Model
from scipy.sparse import csr_matrix

class GraphFactorization(Model):

    def __init__(self, graph: Graph,
                 d: int = 2,
                 similarity_matrix: csr_matrix = None,
                 eps: float = 1e-7,
                 lmbd: float = 0) -> None:
        """
        The initialization method of the Graph Factorization model.
        :param graph: The graph to be embedded
        :param d: dimensionality of the embedding vectors
        :param similarity_matrix: Similarity matrix of the graph. Adjacency matrix of the graph is passed by default
        :param eps: Threshold value of the change in optimisation process.
        The algorithm stops when the difference between two reprezentations is less than eps
        :param lmbd: Regularisation coefficient.
        """
        __A: csr_matrix
        __d: int
        __eps: float
        __lmbd: float

        super().__init__(graph)
        self.__A = similarity_matrix if similarity_matrix is not None else adjacency_matrix(self.get_graph())
        self.__d = d
        self.__eps = eps
        self.__lmbd = lmbd


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

