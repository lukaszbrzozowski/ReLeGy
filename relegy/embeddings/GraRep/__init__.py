from relegy.__base import Model
import numpy as np
import networkx as nx
from networkx import Graph
import tensorflow as tf

construct_verification = {"graph": [(lambda x: type(x) == Graph, "'graph' must be a networkx graph")]}

init_verification = {"lmbd": [(lambda x: x > 0, "'lmbd' must be greater than 0.")]}

fit_verification = {"max_K": [(lambda x: x >= 1, "'max_K' must be at least 1.")],
                    "d": [(lambda x: True if x is None else x > 0, "'d' must be greater than 0.")]}

embed_verification = {"K": [(lambda x: x >= 1, "'K' must be at least 1.")],
                      "d": [(lambda x: True if x is None else x > 0, "'d' must be greater than 0.")]}

fast_embed_verification = Model.dict_union(construct_verification, init_verification, embed_verification)


class GraRep(Model):
    """
    The GraRep method implementation. \n
    The details may be found in: \n
    'S. Cao, W. Lu, and Q. Xu. Grarep: Learning graph representations with global structural information. In KDD, 2015'
    """

    @Model._verify_parameters(rules_dict=construct_verification)
    def __init__(self,
                 graph: Graph,
                 keep_full_SVD: bool = True):
        """
        GraRep - constructor (step I)

        @param graph: The graph to be embedded.
        @param keep_full_SVD: if True, the GraRep instance keeps full SVDs decomposition in memory. This allows for fast
         calculation of embedding after fitting for different values of 'd'. If False, 'd' must be provided during
         fitting and cannot be changed in the 'embed' method.

        """

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

        self.__max_K = 0

    @Model._init_in_init_fit
    @Model._verify_parameters(rules_dict=init_verification)
    def initialize(self,
                   lmbd: float = 1):
        """
        GraRep - initialize (step II) \n
        Generates the adjacency and transition matrices.

        @param lmbd: regularization parameter, as described in the paper.
        """

        graph = self.get_graph()
        self.__lmbd = tf.constant(lmbd, dtype="float32")
        self.__N = len(graph.nodes)
        self.__beta = tf.constant(self.__lmbd / self.__N, dtype="float32")
        self.__A = tf.convert_to_tensor(nx.to_numpy_array(graph, nodelist=np.arange(len(graph.nodes))), dtype="float32")
        self.__D = tf.linalg.diag(tf.pow(tf.reduce_sum(self.__A, 1), -1))
        self.__S = tf.matmul(self.__D, self.__A)
        self.__S_cur = tf.eye(self.__N, dtype="float32")

    @Model._fit_in_init_fit
    @Model._verify_parameters(rules_dict=fit_verification)
    def fit(self,
            max_K=1,
            d=None):
        """
        GraRep - fit (step III) \n
        Calculates the SVD decompositions for matrices determining 1, ..., max_K similarity orders.

        @param max_K: maximal similarity order K for which the SVD decomposition are to be calculated. K in the 'embed'
        must equal or smaller than max_K in the 'fit' method.
        @param d: The embedding dimension. Must be passed when keep_full_SVDs is false, ignored otherwise.
        """

        if max_K > self.__max_K:
            self.__max_K = max_K
        if not self.__keep_SVD:
            if d is None:
                raise Exception("The 'd' parameter cannot be None when 'keep_full_SVD' is false")
            self.__results = []
            for i in range(1, max_K + 1):
                self.__S_cur = tf.matmul(self.__S_cur, self.__S)
                gamma = tf.repeat(tf.reshape(tf.reduce_sum(self.__S_cur, 0), shape=[self.__N, 1]), [self.__N], axis=1)
                X = tf.math.log(self.__S_cur / gamma) - tf.math.log(self.__beta)
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
                for i in range(cur_K + 1, max_K + 1):
                    self.__S_cur = tf.matmul(self.__S_cur, self.__S)
                    gamma = tf.repeat(tf.reshape(tf.reduce_sum(self.__S_cur, 0), shape=[self.__N, 1]), [self.__N],
                                      axis=1)
                    X = tf.math.log(self.__S_cur / gamma) - tf.math.log(self.__beta)
                    X = tf.where(X < 0, tf.zeros_like(X), X)
                    D, U, VT = tf.linalg.svd(X)
                    self.__SVDs.append((D, U, VT))

    @Model._embed_in_init_fit
    @Model._verify_parameters(rules_dict=embed_verification)
    def embed(self,
              K=1,
              d=2,
              concatenated=True):
        """
        GraRep - embed (step IV) \n
        Returns the embedding in the form of a matrix with dimension Nx(K*d).
        @param K: Similarity order of the embedding. K in the 'embed'
        must be equal or smaller than max_K in the 'fit' method.
        @param d: The embedding dimension. Must be passed when keep_full_SVDs is True, ignored otherwise.
        @param concatenated: If False, returns a list of K matrices of dimensions Nxd.
        @return: The embedding matrix/matrices.
        """

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
            # noinspection PyUnresolvedReferences
            return [retList[i].numpy() for i in range(K)]

    @staticmethod
    @Model._verify_parameters(rules_dict=fast_embed_verification)
    def fast_embed(graph: Graph,
                   d: int = 2,
                   K: int = 1,
                   lmbd: float = 1,
                   concatenated: bool = True):
        """
        GraRep - fast_embed \n
        Performs the embedding in a single step. The parameter keep_full_SVDs is passed as False, as the embedding is
        calculated once.
        @param graph: The graph to be embedded. Present in '__init__'
        @param d: The embedding dimension. Present in 'fit'
        @param K: Similarity order of the embedding. Present in 'embed'
        @param lmbd: Regularization parameter, as described in the paper. Present in 'initialize'
        @param concatenated: If False, returns a list of K matrices of dimensions Nxd. Present in 'embed'
        @return: The embedding matrix/matrices.
        """
        GR = GraRep(graph=graph,
                    keep_full_SVD=False)
        GR.initialize(lmbd=lmbd)
        GR.fit(max_K=K,
               d=d)
        return GR.embed(K=K, concatenated=concatenated)
