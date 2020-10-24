from engthesis.model import Model
from numpy import ndarray, arange
from networkx import Graph, to_numpy_array
import tensorflow as tf
import warnings

class GraRep(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 K: int = 1,
                 lmbd: float = 1):

        super().__init__(graph)
        A = to_numpy_array(graph, nodelist=arange(len(graph.nodes)))
        self.__A = tf.constant(A, dtype="float32")
        self.__d = tf.constant(d)
        self.__K = tf.constant(K)
        self.__lmbd = tf.constant(lmbd, dtype="float32")
        self.__N = tf.constant(A.shape[0], dtype="float32")
        self.__tensor_list = None
    def info(self) -> str:
        raise NotImplementedError

    def embed(self) -> ndarray:
        beta = tf.constant(self.__lmbd/self.__N, dtype="float32")
        D = tf.linalg.diag(tf.pow(tf.reduce_sum(self.__A, 1), -1))
        S = tf.matmul(D, self.__A)
        S_cur = tf.eye(self.__N, dtype="float32")
        tensor_list = [None] * self.__K.numpy()
        for i in range(1, self.__K + 1):
            S_cur = tf.matmul(S_cur, S)
            gamma = tf.repeat(tf.reshape(tf.reduce_sum(S_cur, 0), shape=[self.__N, 1]), [self.__N], axis=1)
            X = tf.math.log(S_cur/gamma) - tf.math.log(beta)
            X = tf.where(X < 0, tf.zeros_like(X), X)
            D, U, VT = tf.linalg.svd(X)
            Ud = U[:, :self.__d]
            Dd = tf.linalg.diag(D[:self.__d])
            W = tf.matmul(Ud, tf.sqrt(Dd))
            tensor_list[i-1] = W
        self.__tensor_list = tensor_list
        return tf.concat(tensor_list, 1).numpy()

    def get_matrix_dict(self):
        if self.__tensor_list is None:
            warnings.warn("Embedding before returning the dictionary")
            self.embed()
        tl = self.__tensor_list
        md = {"W1": tl[0]}
        for i in range(1, self.__K):
            md["W" + str(i+1)] = tl[i].numpy()
        return md
