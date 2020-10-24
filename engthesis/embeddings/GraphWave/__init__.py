from engthesis.model import Model
from networkx import Graph, laplacian_matrix
import numpy as np
from numpy import ndarray
import tensorflow as tf


class GraphWave(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 J: int = 1,
                 eta: float = 0.85,
                 gamma: float = 0.95,
                 kernel = lambda x, s: tf.math.exp(-x * s),
                 interval_start: float = 0,
                 interval_stop: float = 1):
        super().__init__(graph)
        self.__d = d
        self.__J = J
        self.__eta = eta
        self.__gamma = gamma
        self.__N = len(graph.nodes)
        self.__L = tf.convert_to_tensor(laplacian_matrix(graph, nodelist=np.arange(self.__N)).toarray(),
                                        dtype="float32")
        self.__kernel = kernel
        self.__i_start = tf.constant(interval_start, "float32")
        self.__i_stop = tf.constant(interval_stop, "float32")

    def info(self) -> str:
        raise NotImplementedError

    def __calculate_theta(self):
        u, v = tf.linalg.eigh(self.__L)
        s = self.__calculate_s(u)
        thetas = [None] * self.__J
        for i in range(self.__J):
            thetas[i] = tf.matmul(v, tf.matmul(tf.linalg.diag(self.__kernel(u, s[i])), tf.transpose(v)))
        return thetas

    def __calculate_s(self, u):
        su = tf.sort(u)
        geom_mean = tf.sqrt(su[1] * su[-1])
        s_max = -tf.math.log(self.__eta) * geom_mean
        s_min = -tf.math.log(self.__gamma) * geom_mean
        if self.__J == 1:
            s = tf.reshape((s_min+s_max)/2, 1)
        else:
            s = tf.linspace(s_min, s_max, self.__J)
        return s

    def embed(self) -> ndarray:
        thetas = tf.cast(self.__calculate_theta(), "complex64")
        t = tf.cast(tf.linspace(self.__i_start, self.__i_stop, self.__d), "complex64")
        Z = np.empty((self.__N, 2*self.__J * self.__d))
        for iter_j in range(self.__J):
            for iter_i in range(t.shape[0]):
                cur_t = t[iter_i]
                phi = tf.reduce_mean(tf.exp(1j * cur_t * thetas[iter_j]), axis=0)
                Z[:, 2*iter_j*self.__d+2*iter_i] = tf.math.real(phi)
                Z[:, 2*iter_j*self.__d+2*iter_i+1] = tf.math.imag(phi)
        return Z