from networkx import to_numpy_matrix, Graph
import numpy as np
from engthesis.model.base import Model
from scipy.sparse import csr_matrix

class GraphFactorization(Model):

    def __init__(self, graph: Graph,
                 d: int = 2,
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
        super().__init__(graph)

        self.__A: csr_matrix = to_numpy_matrix(self.get_graph(), nodelist=np.arange(len(self.get_graph().nodes)))
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

