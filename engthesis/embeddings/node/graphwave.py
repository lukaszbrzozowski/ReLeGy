from engthesis.model.base import Model
from networkx import Graph, laplacian_matrix
import numpy as np
from numpy import ndarray


class GraphWave(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 use_chebyshew: bool = False,
                 K: int = None,
                 J: int = 1,
                 eta: float = 0.85,
                 gamma: float = 0.95,
                 kernel=lambda x, s: np.exp(-x * s),
                 interval_start: float = 0,
                 interval_stop: float = 1):
        super().__init__(graph)
        self.__d = d
        self.__uC = use_chebyshew
        if use_chebyshew:
            assert (K is not None)
        self.__K = K
        self.__J = J
        self.__eta = eta
        self.__gamma = gamma
        self.__L = laplacian_matrix(graph, nodelist=np.arange(len(graph.nodes))).toarray()
        print(self.__L)
        self.__kernel = kernel
        self.__i_start = interval_start
        self.__i_stop = interval_stop

    def calculate_theta(self):
        if self.__uC:
            raise NotImplementedError
        else:
            u, v = np.linalg.eigh(self.__L)
            s = self.calculate_s(u)
            thetas = [None] * len(s)
            for i in range(len(s)):
                thetas[i] = v @ np.diag(self.__kernel(u, s[i])) @ v.T
        return thetas

    def calculate_s(self, u):
        su = np.sort(u)
        geometric_mean = np.sqrt(su[1] * su[-1])
        s_max = -np.log(self.__eta) * geometric_mean
        s_min = -np.log(self.__gamma) * geometric_mean
        if self.__J == 1:
            s = [(s_min + s_max) / 2]
        else:
            s = np.linspace(s_min, s_max, self.__J)
        print(s)
        return s

    def info(self) -> str:
        raise NotImplementedError

    def embed(self) -> ndarray:
        theta = self.calculate_theta()
        t = np.linspace(self.__i_start, self.__i_stop, self.__d)
        N = len(self.get_graph().nodes)
        Z = np.empty((N, 2 * self.__J * self.__d))
        for iter_j in range(self.__J):
            for i in range(t.shape[0]):
                cur_t = t[i]
                phi = np.mean(np.exp(1j * cur_t * theta[iter_j]), axis=0)
                Z[:, 2*iter_j*self.__d+2*i] = np.real(phi)
                Z[:, 2*iter_j*self.__d+2*i+1] = np.imag(phi)
        return Z
