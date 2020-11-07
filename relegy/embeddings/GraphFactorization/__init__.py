from relegy.__base import Model
import numpy as np
import networkx as nx
from numpy import ndarray
from networkx import Graph
import tensorflow as tf


class GraphFactorization(Model):
    """
    The Graph Factorization method implementation. \n
    The details may be found in: \n
    'A. Ahmed, N. Shervashidze,
    S. Narayanamurthy, V. Josifovski, and A.J. Smola. Distributed large-scale natural graph factorization. In WWW,
    2013'
    """

    def __init__(self,
                 graph: Graph):
        """
        Graph Factorization - constructor (step I)

        @param graph: The graph to be embedded.

        """

        self.__A = None
        self.__N = None
        self.__mask = None
        self.__lmbd = None
        self.__d = None
        self.__model = None

        super().__init__(graph)

    def info(self) -> str:
        raise NotImplementedError

    @Model._init_in_init_model_fit
    def initialize(self,
                   d: int = 2,
                   lmbd: float = 0.1):
        """
        Graph Factorization - initialize (step II) \n
        Generates the adjacency matrix.

        @param d: The dimension of the embedding
        @param lmbd: Regularization parameter in the cost function
        """
        graph = self.get_graph()
        A = nx.to_numpy_array(graph, nodelist=np.arange(len(graph.nodes)))
        self.__A = tf.constant(A, dtype="float32")
        self.__N = tf.constant(A.shape[0])
        self.__mask = tf.constant((A > 0), dtype="float32")
        self.__lmbd = tf.constant(lmbd)
        self.__d = tf.constant(d)

    def __get_loss(self, model):
        y_pred = model(tf.eye(self.__N))
        main_loss = 0.5 * tf.reduce_sum(tf.multiply(self.__mask,
                                                    tf.math.pow(self.__A - tf.matmul(y_pred, tf.transpose(y_pred)), 2)))
        reg_loss = self.__lmbd / 2 * tf.pow(tf.norm(y_pred), 2)
        return main_loss + reg_loss

    def __get_gradients(self, model):
        with tf.GradientTape() as tape:
            tape.watch(model.variables)
            L = self.__get_loss(model)
        g = tape.gradient(L, model.variables)
        return g

    @Model._init_model_in_init_model_fit
    def initialize_model(self,
                         optimizer: str = "adam",
                         lr: float = 0.1,
                         verbose: bool = False):
        """
        Graph Factorization - initialize_model (step III) \n
        Generates a neural network to optimize the cost function.
        @param optimizer: The name of the optimizer. Must be one of tf.keras optimizer names.
        @param lr: The learning rate of the model.
        @param verbose: Verbosity parameter.
        """

        input_layer = tf.keras.Input(shape=[self.__N],
                                     batch_size=None)
        output_layer = tf.keras.layers.Dense(self.__d, activation="linear")
        model = tf.keras.Sequential([input_layer, output_layer])
        optimizer_ent = tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": lr}})
        model.compile(optimizer=optimizer_ent)
        self.__model = model
        if verbose:
            print("The model has been built")

    @Model._fit_in_init_model_fit
    def fit(self,
            num_iter: int = 300,
            verbose: bool = True):
        """
        Graph Factorization - fit (step IV) \n
        Performs the neural network fitting to the cost function.

        @param num_iter: The number of iterations of fitting.
        @param verbose: Verbosity parameter.
        """

        model = self.__model
        optimizer = model.optimizer
        for i in range(num_iter):
            g = self.__get_gradients(model)
            optimizer.apply_gradients(zip(g, model.variables))
            if verbose:
                print("Epoch " + str(i + 1) + ": " + str(self.__get_loss(model).numpy()))
        self.__model = model

    @Model._embed_in_init_model_fit
    def embed(self) -> ndarray:
        """
        Graph Factorization - embed (step V) \n
        Returns the embedding.

        @return: The embedding matrix, shape: N x d.
        """
        return self.__model(tf.eye(self.__N)).numpy()

    @staticmethod
    def fast_embed(graph: Graph,
                   d: int = 2,
                   lmbd: float = 0.1,
                   optimizer: str = "adam",
                   lr: float = 0.1,
                   init_model_verbose=True,
                   num_iter: int = 300,
                   fit_verbose: bool = True):
        """
        Graph Factorization - fast_embed \n

        Performs the embedding in one step.

        @param graph: The graph to be embedded. Present in '__init__'
        @param d: The dimension of the embedding. Present in 'initialize'
        @param lmbd: Regularization parameter in the cost function. Present in 'initialize'
        @param optimizer: The name of the optimizer. Must be one of tf.keras optimizer names. Present in 'initialize_model'
        @param lr: The learning rate of the model. Present in 'initialize_model'
        @param init_model_verbose: Verbosity parameter. Present in 'initialize_model'
        @param num_iter: The number of iterations of fitting. Present in 'fit'
        @param fit_verbose: Verbosity parameter. Present in 'fit'
        @return: The embedding matrix, shape: N x d.
        """
        GF = GraphFactorization(graph)
        GF.initialize(d=d,
                      lmbd=lmbd)
        GF.initialize_model(optimizer=optimizer,
                            lr=lr,
                            verbose=init_model_verbose)
        GF.fit(num_iter=num_iter,
               verbose=fit_verbose)
        return GF.embed()

    def get_loss(self):
        """
        Graph Factorization - get_loss
        Returns the value of the loss function of the current network.
        @return: Value of the loss function.
        """
        if not self._initialized:
            raise Exception(
                "The methods 'initialize' and 'initialize_model' must be called before evaluating the __base")
        if not self._initialized_model:
            raise Exception("The method 'initialize_model' must be called before evaluating the __base")

        return self.__get_loss(self.__model)
