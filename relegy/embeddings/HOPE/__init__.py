from relegy.__base import Model
import networkx as nx
from networkx import Graph, DiGraph
import numpy as np

import tensorflow as tf

construct_verification = {"graph": [(lambda x: type(x) == Graph or type(x) == DiGraph, "'graph' must be a networkx Graph or DiGraph")]}

universal_verification = {"d": [(lambda x: True if x is None else x > 0, "'d' must be greater than 0.")]}



class HOPE(Model):
    """
    The HOPE method implementation. \n
    The details may be found in: \n
    'M. Ou, P. Cui, J. Pei, Z. Zhang, and W. Zhu. Asymmetric transitivity preserving graph embedding. In KDD, 2016.'
    """

    @Model._verify_parameters(rules_dict=construct_verification)
    def __init__(self,
                 graph: Graph,
                 keep_full_SVD: bool = True):
        """
        HOPE - constructor (step I)

        @param graph: The graph to be embedded. Nodes of the graph must be a sorted array from 0 to n-1, where n is
        the number of vertices. May be weighted and/or directed.
        @param keep_full_SVD: if True, the HOPE instance keeps full SVDs decomposition in memory. This allows for fast
         calculation of embedding after fitting for different values of 'd'. If False, 'd' must be provided during
         fitting and cannot be changed in the 'embed' method.

        """

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
        """
        HOPE - initialize (step II) \n
        Generates the adjacency matrix and proximity measure. The measures and parameters are described in the HOPE
        article.

        @param proximity: Name of the proximity measure. Possible are 'Katz' - Katz Index, 'RPR' - Rooted PageRank,
        'CN' - Common Neighbours and 'AA' - Adamic-Adar.
        @param kwargs: Parameter 'beta' of 'Katz' proximity measure or parameter 'alpha' of 'RPR' proximity measure
        may be given here. Assumed 'beta' or 'alpha' equal to 0.1 otherwise. Ignored, if 'AA' or 'CN' are used.
        """
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
        """
        Calculation of Katz Index matrix decomposition.
        """
        Mg = tf.eye(self.__N) - self.__proximity_param*self.__A
        Ml = self.__proximity_param*self.__A
        return Mg, Ml

    def __proximity_rpr(self):
        """
        Calculation of RPR matrix decomposition.
        """
        D = tf.linalg.diag(tf.pow(tf.reduce_sum(self.__A, axis=1), -1))
        P = tf.matmul(D, self.__A)
        Mg = tf.eye(self.__N) - self.__proximity_param * P
        Ml = (1-self.__proximity_param) * tf.eye(self.__N)
        return Mg, Ml

    def __proximity_cn(self):
        """
        Calculation of CN matrix decomposition.
        """
        Mg = tf.eye(self.__N)
        Ml = tf.matmul(self.__A, self.__A)
        return Mg, Ml

    def __proximity_aa(self):
        """
        Calculation of AA matrix decomposition.
        """
        row_sums = tf.reduce_sum(self.__A, 1)
        col_sums = tf.reduce_sum(self.__A, 0)
        D = tf.linalg.diag(tf.pow(row_sums+col_sums, -1))
        Mg = tf.eye(self.__N)
        Ml = tf.matmul(self.__A, tf.matmul(D, self.__A))
        return Mg, Ml

    def __get_prox(self):
        """
        Returns the proper proximity measure matrix decomposition.
        @return:
        """
        if self.__proximity == "Katz":
            return self.__proximity_katz()
        elif self.__proximity == "RPR":
            return self.__proximity_rpr()
        elif self.__proximity == "CN":
            return self.__proximity_cn()
        else:
            return self.__proximity_aa()

    @Model._fit_in_init_fit
    @Model._verify_parameters(rules_dict=universal_verification)
    def fit(self,
            d=None):
        """
        HOPE - fit (step III) \n
        Calculates the SVD decompositions of the two embedding matrices.

        @param d: The embedding dimension. Must be passed if keep_full_SVDs is False, ignored otherwise.
        """

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
    @Model._verify_parameters(rules_dict=universal_verification)
    def embed(self,
              d=2,
              concatenated=True):
        """
        HOPE - embed (step IV) \n
        returns the embedding matrix.

        @param d: The embedding dimension. Must be passed if keep_full_SVDs is True, ignored otherwise.
        @param concatenated: The result of the HOPE method are two matrices: Us and Ut. If 'concatenated' is True,
        returns the concatenation of the matrices, that is a Nx(2*d) matrix. If False, returns the matrix multiplication
        of Us^T and Ut, as described in the article.
        @return: The embedding matrix.
        """
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
            return tf.matmul(Us, tf.transpose(Ut)).numpy()

    @staticmethod
    @Model._verify_parameters(rules_dict=universal_verification)
    def fast_embed(graph: Graph,
                   proximity: str = "Katz",
                   d: int = 2,
                   concatenated=True,
                   **kwargs):
        """
        HOPE - fast_embed. \n
        Returns the embedding in a single step. The parameter keep_full_SVDs is passed as False, as the embedding is
        calculated once.
        @param graph: The graph to be embedded. Present in '__init__'
        @param proximity: Name of the proximity measure. Possible are 'Katz' - Katz Index, 'RPR' - Rooted PageRank,
        'CN' - Common Neighbours and 'AA' - Adamic-Adar. Present in 'initialize'
        @param d: The embedding dimension. Present in 'fit'
        @param concatenated: The result of the HOPE method are two matrices: Us and Ut. If 'concatenated' is True,
        returns the concatenation of the matrices, that is a Nx(2*d) matrix. If False, returns the matrix multiplication
        of Us^T and Ut, as described in the article. Present in 'embed'
        @param kwargs: Parameter 'beta' of 'Katz' proximity measure or parameter 'alpha' of 'RPR' proximity measure
        may be given here. Assumed 'beta' or 'alpha' equal to 0.1 otherwise. Ignored, if 'AA' or 'CN' are used.
        Present in 'initialize'
        @return: The embedding matrix.
        """
        hope = HOPE(graph, keep_full_SVD=False)
        hope.initialize(proximity=proximity,
                        kwargs=kwargs)
        hope.fit(d=d)
        return hope.embed(concatenated=concatenated)


