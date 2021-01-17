import numpy as np
import networkx as nx
from networkx import Graph
from relegy.__base import Model
import tensorflow as tf

construct_verification = {"graph": [(lambda x: issubclass(type(x), Graph), "'graph' must be a networkx graph")]}

init_verification = {"d": [(lambda d: d > 0, "d has to be greater than 0.")]}

init_model_verification = {"lmbd1": [(lambda x: x >= 0, "lmbd1 must be non-negative.")],
                           "lmbd2": [(lambda x: x >= 0, "lmbd2 must be non-negative.")],
                           "lr1": [(lambda x: x > 0, "lr1 must be greater than 0.")],
                           "lr2": [(lambda x: x > 0, "lr2 must be greater than 0.")]}

fit_verification = {"num_iter": [(lambda x: x > 0, "num_iter must be greater than 0,")]}

fast_embed_verification = Model.dict_union(construct_verification, init_verification, init_model_verification, fit_verification)


class LINE(Model):
    """
    The LINE method implementation. \n
    The details may be found in: \n
    ' J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei. Line: Large-scale information network embedding. In
    WWW, 2015.'
    """
    @Model._verify_parameters(rules_dict=construct_verification)
    def __init__(self,
                 graph: Graph):
        """
        LINE - constructor (step I)
        @param graph: The graph to be embedded. Nodes of the graph must be a sorted array from 0 to n-1, where n is
        the number of vertices.
        """
        super().__init__(graph.to_directed())
        self.__d = None
        self.__lr1 = None
        self.__lr2 = None
        self.__lmbd1 = None
        self.__lmbd2 = None
        self.__opt1 = None
        self.__opt2 = None
        self.__A = None
        self.__U1 = None
        self.__U2 = None
        self.__N = None

    @Model._init_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_verification)
    def initialize(self,
                   d=2):
        """
        LINE - initialize (step II) \n
        Generation of adjacency matrix and input matrices.
        @param d: The embedding dimension.
        """
        self.__N = len(self.get_graph().nodes)
        self.__d = d
        self.__A = tf.convert_to_tensor(nx.to_numpy_array(self.get_graph(), nodelist=np.arange(self.__N)), dtype="float32")
        self.__U1 = tf.Variable(tf.random.uniform([self.__N, d]))
        self.__U2 = tf.Variable(tf.random.uniform([self.__N, d]))

    def __get_loss1(self):
        return -tf.reduce_sum(tf.multiply(self.__A,
                                          tf.math.log(tf.divide(1,
                                                                1+tf.math.exp(tf.matmul(-self.__U1,
                                                                                        tf.transpose(self.__U2))))))) + tf.reduce_sum(self.__lmbd1 * (tf.abs(self.__U1)+tf.abs(self.__U2)))

    def __get_loss2(self):
        d_temp = tf.reduce_sum(self.__A, axis=1)
        d = tf.tile(tf.reshape(d_temp, [self.__N, 1]), tf.constant([1, self.__N]))
        mlog = tf.math.log(tf.divide(self.__A, d))
        return -tf.reduce_sum(tf.math.multiply_no_nan(mlog, self.__A)) + tf.reduce_sum(self.__lmbd2 * (tf.abs(self.__U1)+tf.abs(self.__U2)))

    def __get_gradients1(self):
        with tf.GradientTape() as tape:
            tape.watch([self.__U1, self.__U2])
            L = self.__get_loss1()
        g = tape.gradient(L, [self.__U1, self.__U2])
        return g

    def __get_gradients2(self):
        with tf.GradientTape() as tape:
            tape.watch([self.__U1, self.__U2])
            L = self.__get_loss2()
        g = tape.gradient(L, [self.__U1, self.__U2])
        return g

    @Model._init_model_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_model_verification)
    def initialize_model(self,
                         optimizer1="adam",
                         optimizer2="adam",
                         lmbd1=0.01,
                         lmbd2=0.01,
                         lr1=0.01,
                         lr2=0.1):
        """
        LINE - initialize_model (step III) \n
        Sets the learning rate and regularization parameters values.

        @param lmbd1: Regularization parameter for the first loss function.
        @param lmbd2: Regularization parameter for the second loss function.
        @param lr1: Learning rate for the the first loss function optimization.
        @param lr2: Learning rate for the the second loss function optimization.
        @param optimizer1: Name of the optimizer for the first loss function. Must be one of keras optimizers.
        @param optimizer1: Name of the optimizer for the second loss function. Must be one of keras optimizers.
        """
        self.__lmbd1 = lmbd1
        self.__lmbd2 = lmbd2
        self.__lr1 = lr1
        self.__lr2 = lr2
        self.__opt1 = tf.keras.optimizers.get({"class_name": optimizer1, "config": {"learning_rate": lr1}})
        self.__opt2 = tf.keras.optimizers.get({"class_name": optimizer2, "config": {"learning_rate": lr2}})

    @Model._fit_in_init_model_fit
    @Model._verify_parameters(rules_dict=fit_verification)
    def fit(self,
            num_iter: int = 300,
            verbose: bool = True):
        """
        LINE - fit (step IV) \n
        Performs the fitting to both loss functions.
        @param num_iter: The number of iterations of the fitting process.
        @param verbose: Verbosity parameter.
        """
        for i in range(num_iter):
            g1 = self.__get_gradients1()
            self.__opt1.apply_gradients(zip(g1, [self.__U1, self.__U2]))
            g2 = self.__get_gradients1()
            self.__opt1.apply_gradients(zip(g2, [self.__U1, self.__U2]))
            if verbose:
                print("Epoch " + str(i + 1) + ": loss 1: " + str(self.__get_loss1().numpy()) + ", loss 2:" + str(self.__get_loss2().numpy()))

    @Model._embed_in_init_model_fit
    @Model._verify_parameters(rules_dict={})
    def embed(self):
        """
        LINE - embed (step V) \n
        Returns the embedding as a concatenation of two matrices minimizing the two loss functions.
        @return: The embedding matrix having shape Nx(2*d).
        """
        return np.concatenate((self.__U1.numpy(), self.__U2.numpy()), axis=1)

    @staticmethod
    @Model._verify_parameters(rules_dict=fast_embed_verification)
    def fast_embed(graph: Graph,
                   d: int = 2,
                   optimizer1: str = "adam",
                   optimizer2: str = "adam",
                   lmbd1: float = 0.01,
                   lmbd2: float = 0.01,
                   lr1: float = 0.01,
                   lr2: float = 0.1,
                   num_iter: int = 300,
                   fit_verbose: bool = True):
        """
        LINE - fast_embed \n
        Returns the embedding in a single step.

        @param graph: The graph to be embedded. Present in '__init__'
        @param d: The embedding dimension. Present in 'initialize'
        @param optimizer1: Name of the optimizer for the first loss function. Must be one of keras optimizers. Present in 'initialize_model'
        @param optimizer1: Name of the optimizer for the second loss function. Must be one of keras optimizers. Present in 'initialize_model'
        @param lmbd1: Regularization parameter for the first loss function. Present in 'initialize_model'
        @param lmbd2: Regularization parameter for the second loss function. Present in 'initialize_model'
        @param lr1: Learning rate for the the first loss function optimization. Present in 'initialize_model'
        @param lr2: Learning rate for the the second loss function optimization. Present in 'initialize_model'
        @param num_iter: The number of iterations of the fitting process. Present in 'fit'
        @param fit_verbose: Verbosity parameter. Present in 'fit'
        @return: The embedding matrix having shape Nx(2*d).
        """
        line = LINE(graph)
        line.initialize(d)
        line.initialize_model(optimizer1,
                              optimizer2,
                              lmbd1,
                              lmbd2,
                              lr1,
                              lr2)
        line.fit(num_iter,
                 fit_verbose)
        return line.embed()