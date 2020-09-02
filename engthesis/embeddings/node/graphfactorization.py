from networkx import adjacency_matrix
import numpy as np
from engthesis.model.base import Model
from scipy.sparse import csr_matrix

class GraphFactorization(Model):

    def __init__(self, graph, **kwargs) -> None:
        """

        :rtype: object
        """
        __A: csr_matrix
        __d: int
        __eps: float
        __lmbd: float

        super().__init__(graph)
        parameters = kwargs
        self.__A = parameters["A"] if "A" in parameters else adjacency_matrix(self.get_graph(),
                                                                              nodelist=np.arange(len(self.get_graph().nodes)))
        self.__d = parameters["d"] if "d" in parameters else 2
        self.__eps = parameters["eps"] if "eps" in parameters else 1e-7
        self.__lmbd = parameters["lmbd"] if "lmbd" in parameters else 0

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

