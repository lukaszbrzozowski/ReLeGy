from relegy.__base import Model
import networkx as nx
from networkx import Graph
import numpy as np
from numpy import ndarray
import tensorflow as tf


class GraphWave(Model):
    """
    The GraphWave method implementation. \n
    The details may be found in: \n
    'C. Donnat, M. Zitnik, D. Hallac, and J. Leskovec. Learning structural node embeddings via diffusion wavelets. arXiv
preprint arXiv:1710.10321, 2017.'
    """

    def __init__(self,
                 graph: Graph):
        """
        GraphWave - constructor (step I)

        @param graph: The graph to be embedded.
        """
        super().__init__(graph)
        self.__J = None
        self.__eta = None
        self.__gamma = None
        self.__N = None
        self.__L = None
        self.__kernel = None
        self.__thetas = None
        self.__Z = None

    @Model._init_in_init_fit
    def initialize(self,
                   J: int = 1,
                   eta: float = 0.85,
                   gamma: float = 0.95,
                   kernel = lambda x, s: tf.math.exp(-x * s)):
        """
        GraphWave - initialize (step II) \n
        Calculates optimal kernel parameter and characteristic functions.

        @param J: number of different kernel parameter values, as described in the article.
        @param eta: kernel optimization parameter, as described in the article.
        @param gamma: kernel optimization parameter, as described in the article.
        @param kernel: kernel function of x and s using Tensorflow operations.
        """
        graph = self.get_graph()
        self.__J = J
        self.__eta = eta
        self.__gamma = gamma
        self.__N = len(graph.nodes)
        self.__L = tf.convert_to_tensor(nx.laplacian_matrix(graph, nodelist=np.arange(self.__N)).toarray(),
                                        dtype="float32")
        self.__kernel = kernel
        self.__thetas = tf.cast(self.__calculate_theta(), "complex64")

    @Model._fit_in_init_fit
    def fit(self,
            d: int = 2,
            interval_start: float = 0,
            interval_stop: float = 1):
        """
        GraphWave - fit (step III) \n
        Generates the embedding.

        @param d: The embedding dimension.
        @param interval_start: Start of the interval from which points are chosen, as described in the article.
        @param interval_stop: End of the interval from which points are chosen, as described in the article.
        """
        t = tf.cast(tf.linspace(tf.constant(interval_start, "float32"),
                                tf.constant(interval_stop, "float32"),
                                d),
                    "complex64")
        Z = np.empty((self.__N, 2*self.__J * d))
        for iter_j in range(self.__J):
            for iter_i in range(t.shape[0]):
                cur_t = t[iter_i]
                phi = tf.reduce_mean(tf.exp(1j * cur_t * self.__thetas[iter_j]), axis=0)
                Z[:, 2*iter_j*d+2*iter_i] = tf.math.real(phi)
                Z[:, 2*iter_j*d+2*iter_i+1] = tf.math.imag(phi)
        self.__Z = Z

    @Model._embed_in_init_fit
    def embed(self) -> ndarray:
        """
        GraphWave - embed (step IV) \n
        Returns the embedding.
        @return: The embedding matrix with shape N x (2*J*d)
        """
        return self.__Z

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

    @staticmethod
    def fast_embed(graph: Graph,
                   J: int = 1,
                   eta: float = 0.85,
                   gamma: float = 0.95,
                   kernel=lambda x, s: tf.math.exp(-x * s),
                   interval_start: float = 0,
                   interval_stop: float = 1,
                   d: int = 2) -> ndarray:
        """
        GraphWave - fast_embed \n
        Performs the embedding in a single step.

        @param graph: The graph to be embedded. Present in '__init__'
        @param J: number of different kernel parameter values, as described in the article. Present in 'initialize'
        @param eta: kernel optimization parameter, as described in the article. Present in 'initialize'
        @param gamma: kernel optimization parameter, as described in the article. Present in 'initialize'
        @param kernel: kernel function of x and s using Tensorflow operations. Present in 'initialize'
        @param d: The embedding dimension. Present in 'fit'
        @param interval_start: Start of the interval from which points are chosen, as described in the article. Present
        in 'fit'
        @param interval_stop: End of the interval from which points are chosen, as described in the article. Present in
        'fit'
        @return: The embedding matrix with shape N x (2*J*d)
        """
        gw = GraphWave(graph)
        gw.initialize(J=J,
                      eta=eta,
                      gamma=gamma,
                      kernel=kernel)
        gw.fit(interval_start=interval_start,
               interval_stop=interval_stop,
               d=d)
        return gw.embed()