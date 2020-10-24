from engthesis.model import Model
from numpy import ndarray, arange
from networkx import Graph, to_numpy_array
import tensorflow as tf
import warnings

class GraRep(Model):

    def __init__(self,
                 graph: Graph,
                 keep_full_SVD: bool = True):

        super().__init__(graph)

        self.__A = None
        self.__lmbd = None
        self.__N = None
        self.__tensor_list = None
        self.__D = None
        self.__S = None
        self.__S_cur = None
        self.__K = None
        self.__beta = None
        self.__keep_SVD = keep_full_SVD
        self.__SVDs = []
        self.__results = []

        self.__initialized = False
        self.__fitted = False
        self.__max_K = 0

    def initialize(self,
                   lmbd: float = 1):

        graph = self.get_graph()
        self.__lmbd = tf.constant(lmbd, dtype="float32")
        self.__N = len(graph.nodes)
        self.__beta = tf.constant(self.__lmbd/self.__N, dtype="float32")
        self.__A = tf.convert_to_tensor(to_numpy_array(graph, nodelist=arange(len(graph.nodes))), dtype="float32")
        self.__D = tf.linalg.diag(tf.pow(tf.reduce_sum(self.__A, 1), -1))
        self.__S = tf.matmul(self.__D, self.__A)
        self.__S_cur = tf.eye(self.__N, dtype="float32")

        self.__initialized = True
        self.__fitted = False


    def fit(self,
            max_K=1,
            d=None):

        if max_K > self.__max_K:
            self.__max_K = max_K
        if not self.__keep_SVD:
            if d is None:
                raise Exception("The 'd' parameter cannot be None when 'keep_full_SVD' is false")
            self.__results = []
            for i in range(1, max_K+1):
                self.__S_cur = tf.matmul(self.__S_cur, self.__S)
                gamma = tf.repeat(tf.reshape(tf.reduce_sum(self.__S_cur, 0), shape=[self.__N, 1]), [self.__N], axis=1)
                X = tf.math.log(self.__S_cur/gamma) - tf.math.log(self.__beta)
                X = tf.where(X < 0, tf.zeros_like(X), X)
                D, U, VT = tf.linalg.svd(X)
                Ud = U[:, :d]
                Dd = tf.linalg.diag(D[:d])
                W = tf.matmul(Ud, tf.sqrt(Dd))
                self.__results.append(W)
        else:
            cur_K = len(self.__SVDs)
            if cur_K >= max_K:
                return
            else:
                for i in range(cur_K+1, max_K+1):
                    self.__S_cur = tf.matmul(self.__S_cur, self.__S)
                    gamma = tf.repeat(tf.reshape(tf.reduce_sum(self.__S_cur, 0), shape=[self.__N, 1]), [self.__N],
                                      axis=1)
                    X = tf.math.log(self.__S_cur / gamma) - tf.math.log(self.__beta)
                    X = tf.where(X < 0, tf.zeros_like(X), X)
                    D, U, VT = tf.linalg.svd(X)
                    self.__SVDs.append((D, U, VT))

    def embed(self,
              K=1,
              d=2,
              concatenated=True):

        if K > self.__max_K:
            raise Exception("The method has been fitted with smaller K")

        if not self.__keep_SVD:
            if concatenated:
                return tf.concat(self.__results[:K], 1).numpy()
            else:
                return [self.__results[i].numpy() for i in range(K)]
        retList = [None] * K
        for i in range(K):
            D, U, VT = self.__SVDs[i]
            Ud = U[:, :d]
            Dd = tf.linalg.diag(D[:d])
            W = tf.matmul(Ud, tf.sqrt(Dd))
            retList[i] = W
        if concatenated:
            return tf.concat(retList, 1).numpy()
        else:
            return [retList[i].numpy() for i in range(K)]

    def info(self) -> str:
        raise NotImplementedError

    @staticmethod
    def fast_embed(graph: Graph,
                   d: int = 2,
                   K: int = 1,
                   lmbd: float = 1,
                   concatenated: bool = True):
        GR = GraRep(graph=graph,
                    keep_full_SVD=False)
        GR.initialize(lmbd=lmbd)
        GR.fit(max_K=K,
               d=d)
        return GR.embed(K=K, concatenated=concatenated)

