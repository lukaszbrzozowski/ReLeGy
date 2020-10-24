from engthesis.model import Model
from networkx import to_numpy_array, Graph
from numpy import ndarray, arange
import tensorflow as tf
import warnings


class HOPE(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 proximity: str = "Katz",
                 **kwargs):

        super().__init__(graph)
        assert proximity in ["Katz", "RPR", "CN", "AA"], "Proximity measure must be 'Katz', 'RPR', 'CN' or 'AA'"
        A = to_numpy_array(graph, nodelist=arange(len(graph.nodes)))
        self.__N = tf.constant(A.shape[0], dtype="float32")
        self.__A = tf.constant(A)
        self.__proximity = proximity
        self.__proximity_param = None
        self.__d = tf.constant(d)
        if proximity == "Katz":
            self.__proximity_param = tf.constant(kwargs["beta"]) if "beta" in kwargs else tf.constant(0.1)
        elif proximity == "RPR":
            self.__proximity_param = tf.constant(kwargs["alpha"]) if "alpha" in kwargs else tf.constant(0.1)

        self.__Us = None
        self.__Ut = None

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

    def embed(self) -> ndarray:
        Mg, Ml = self.__get_prox()
        S = tf.matmul(tf.linalg.inv(Mg), Ml)
        D, U, VT = tf.linalg.svd(S)

        Ds = tf.linalg.diag(tf.sqrt(D[:self.__d]))
        Us = tf.matmul(U[:, :self.__d], Ds)
        Ut = tf.matmul(tf.transpose(VT)[:, :self.__d], Ds)
        self.__Us = Us
        self.__Ut = Ut
        return tf.matmul(tf.transpose(Us), Ut).numpy()

    def get_matrices(self):
        if self.__Us is None or self.__Ut is None:
            warnings.warn("Embedding before returning the dictionary")
            self.embed()
        return {"Us": self.__Us, "Ut": self.__Ut}


