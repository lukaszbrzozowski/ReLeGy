import numpy as np
from engthesis.model.base import Model
from networkx import Graph, diameter, floyd_warshall_numpy
from numpy import ndarray
from fastdtw import fastdtw

class Struct2Vec(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 T: int = 40,
                 gamma: int = 1,
                 window_size: int = 5):
        """

        :param graph:
        :param d:
        :param T:
        :param gamma:
        :param window_size:
        """
        super().__init__(graph)
        self.__d = d
        self.__T = T
        self.__gamma = gamma
        self.__window = window_size

    def generate_similarity_matrices(self):
        N = len(self.get_graph().nodes)
        deg_seq = np.array(self.get_graph().degree(np.arange(N)))[:, 1].reshape(N, -1)
        k_max = diameter(self.get_graph())
        dist_matrix = floyd_warshall_numpy(self.get_graph())
        matrix_dict = {}
        f_cur = np.zeros((N, N))
        for k in np.arange(1, k_max):
            for u in np.arange(N):
                for v in np.arange(N):
                    mask_u = (dist_matrix[u, :] == k).reshape(N, -1)
                    mask_v = (dist_matrix[v, :] == k).reshape(N, -1)
                    if np.any(mask_u) and np.any(mask_v):
                        deg_u = deg_seq[mask_u]
                        deg_v = deg_seq[mask_v]
                        dist = self.__compare_deg_seq(deg_u, deg_v)
                        f_cur[u, v] += dist
                    else:
                        f_cur[u, v] = 0
            matrix_dict["W" + str(k)] = np.copy(f_cur)
        return matrix_dict



    @staticmethod
    def __compare_deg_seq(deg1, deg2):
        dist_, _ = fastdtw(np.sort(deg1), np.sort(deg2), dist=lambda a, b: max(a, b)/min(a, b) - 1)
        return dist_

    def info(self) -> str:
        raise NotImplementedError

    def embed(self) -> ndarray:
        raise NotImplementedError
