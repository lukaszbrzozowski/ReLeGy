from engthesis.model import Model
import networkx as nx
from networkx import Graph
import numpy as np

import tensorflow as tf


class HOPE(Model):

    def __init__(self,
                 graph: Graph,
                 keep_full_SVD: bool = True):

        super().__init__(graph)
        self.__A = None
        self.__N = None
        self.__proximity = None
        self.__proximity_param = None
        self.__keep_SVD = keep_full_SVD
        self.__Mg = None
        self.__Ml = None
        self.__SVDs = None
        self.__results = None

    @Model._init_in_init_fit
    def initialize(self,
                   proximity: str = "Katz",
                   **kwargs):

        assert proximity in ["Katz", "RPR", "CN", "AA"], "Proximity measure must be 'Katz', 'RPR', 'CN' or 'AA'"
        graph = self.get_graph()
        A = nx.to_numpy_array(graph, nodelist=np.arange(len(graph.nodes)))
        self.__A = tf.constant(A, dtype="float32")
        self.__N = tf.constant(A.shape[0], dtype="float32")
        self.__proximity = proximity
        if proximity == "Katz":
            self.__proximity_param = tf.constant(kwargs["beta"]) if "beta" in kwargs else tf.constant(0.1)
        elif proximity == "RPR":
            self.__proximity_param = tf.constant(kwargs["alpha"]) if "alpha" in kwargs else tf.constant(0.1)

    def __proximity_katz(self):
        Mg = tf.eye(self.__N) - self.__proximity_param*self.__A
        Ml = self.__proximity_param*self.__A
        return Mg, Ml

    def __proximity_rpr(self):
        D = tf.linalg.diag(tf.pow(tf.reduce_sum(self.__A, axis=1), -1))
        P = tf.matmul(D, self.__A)
        Mg = tf.eye(self.__N) - self.__proximity_param * P
        Ml = (1-self.__proximity_param) * tf.eye(self.__N)
        return Mg, Ml

    def __proximity_cn(self):
        Mg = tf.eye(self.__N)
        Ml = tf.matmul(self.__A, self.__A)
        return Mg, Ml

    def __proximity_aa(self):
        row_sums = tf.reduce_sum(self.__A, 1)
        col_sums = tf.reduce_sum(self.__A, 0)
        D = tf.linalg.diag(tf.pow(row_sums+col_sums, -1))
        Mg = tf.eye(self.__N)
        Ml = tf.matmul(self.__A, tf.matmul(D, self.__A))
        return Mg, Ml

    def __get_prox(self):
        if self.__proximity == "Katz":
            return self.__proximity_katz()
        elif self.__proximity == "RPR":
            return self.__proximity_rpr()
        elif self.__proximity == "CN":
            return self.__proximity_cn()
        else:
            return self.__proximity_aa()

    def info(self) -> str:
        raise NotImplementedError

    @Model._fit_in_init_fit
    def fit(self,
            d=None):

        if not self.__keep_SVD:
            if d is None:
                raise Exception("The 'd' parameter cannot be None when 'keep_full_SVD' is false")
            Mg, Ml = self.__get_prox()
            S = tf.matmul(tf.linalg.inv(Mg), Ml)
            D, U, VT = tf.linalg.svd(S)
            Ds = tf.linalg.diag(tf.sqrt(D[:d]))
            Us = tf.matmul(U[:, :d], Ds)
            Ut = tf.matmul(tf.transpose(VT)[:, :d], Ds)
            self.__results = [Us, Ut]
        else:
            Mg, Ml = self.__get_prox()
            S = tf.matmul(tf.linalg.inv(Mg), Ml)
            D, U, VT = tf.linalg.svd(S)
            self.__SVDs = (D, U, VT)

    @Model._embed_in_init_fit
    def embed(self,
              d=2,
              concatenated=True):
        if not self.__keep_SVD:
            if concatenated:
                return tf.concat(self.__results, 1).numpy()
            else:
                return tf.matmul(tf.transpose(self.__results[0]), self.__results[1]).numpy()
        D, U, VT = self.__SVDs
        Ds = tf.linalg.diag(tf.sqrt(D[:d]))
        Us = tf.matmul(U[:, :d], Ds)
        Ut = tf.matmul(tf.transpose(VT)[:, :d], Ds)
        if concatenated:
            return tf.concat([Us, Ut], 1).numpy()
        else:
            return tf.matmul(tf.transpose(Us), Ut).numpy()

    @staticmethod
    def fast_embed(graph: Graph,
                   proximity: str = "Katz",
                   d: int = 2,
                   concatenated=True,
                   **kwargs):
        hope = HOPE(graph, keep_full_SVD=False)
        hope.initialize(proximity=proximity,
                        kwargs=kwargs)
        hope.fit(d=d)
        return hope.embed(concatenated=concatenated)


