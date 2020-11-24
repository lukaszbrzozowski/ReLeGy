from relegy.__base import Model
import networkx as nx
from networkx import Graph
import numpy as np
from numpy import ndarray
import tensorflow as tf
import scipy.sparse as sps
from tensorflow.keras.layers import Layer

init_model_verification = {"n_hid": [(lambda x: np.all(np.array(x) > 0), "Every element of 'n_hid' must be greater than 0.")],
                           "lr": [(lambda x: x > 0, "'lr' must be greater than 0.")]}

fit_verification = {"num_iter": [(lambda x: x > 0, "num_iter must be greater than 0")]}

fast_embed_verification = Model.dict_union(init_model_verification, fit_verification)

class GCN(Model):
    """
    The GCN method implementation. \n
    The details may be found in: \n
    'T.N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In ICLR, 2016.'
    """

    def __init__(self,
                 graph: Graph
                 ):
        """
        GCN - constructor (step I) \n

        @param graph: The graph to be embedded.
        """
        super().__init__(graph)
        self.__X = None
        self.__Y = None
        self.__d = None
        self.__Y_mask = None
        self.__A = None
        self.__A_weighted = None
        self.__N = None
        self.__model = None
        self.__optimizer = None

    class GCNLayer(Layer):
        def __init__(self, output_dim, A_matrix, activation_id, **kwargs):
            self.output_dim = output_dim
            self.__A_matrix = A_matrix
            self.activation_id = activation_id
            self.kernel = None
            self.bias = None
            super(GCN.GCNLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.kernel = self.add_weight(name="kernel",
                                          shape=(input_shape[1], self.output_dim),
                                          initializer="normal", trainable=True, dtype="float32")
            self.bias = self.add_weight(name="bias",
                                        shape=self.output_dim,
                                        initializer="zeros", trainable=True)
            super().build(input_shape)

        def call(self, input_data, **kwargs):
            act = tf.keras.activations.get(self.activation_id)
            return act(tf.matmul(self.__A_matrix, tf.matmul(input_data, self.kernel)) + self.bias)

    @Model._init_in_init_model_fit
    def initialize(self,
                   Y: ndarray,
                   Y_mask: ndarray = None,
                   X: ndarray = None):
        """
        GCN - initialize (step II) \n
        Generates the matrices used in convolution.

        @param Y: The vector of labels for vertices.
        @param Y_mask: A binary vector masking the vertices from loss function calculation. Vertices corresponding to 1
        are included in loss function calculation, vertices corresponding to 0 are not.
        @param X: Matrix of features of vertices. Identity matrix passed if X is None.
        """

        graph = self.get_graph()

        self.__d = len(np.unique(Y))

        self.__A = nx.adjacency_matrix(graph, nodelist=np.arange(len(graph.nodes)))
        self.__N = self.__A.shape[0]

        if X is None:
            self.__X = sps.identity(self.__N)
        else:
            self.__X = X
        if sps.issparse(self.__X):
            self.__X = tf.convert_to_tensor(self.__X.toarray().astype(np.float32))
        else:
            self.__X = tf.convert_to_tensor(self.__X.astype(np.float32))

        self.__Y = Y
        if Y_mask is None:
            self.__Y_mask = np.repeat(1, len(Y))
        else:
            self.__Y_mask = Y_mask

        A_hat = self.__A + sps.identity(self.__A.shape[0])
        D = sps.diags(A_hat.sum(axis=0).A1)
        D_sqrt_inv = D.sqrt().power(-1)
        result = D_sqrt_inv @ A_hat @ D_sqrt_inv

        self.__A_weighted = tf.convert_to_tensor(result.toarray().astype(np.float32))

    @Model._init_model_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_model_verification)
    def initialize_model(self,
                         n_hid = None,
                         activations = None,
                         lr: float = 0.01):
        """
        GCN - initialize_model (step III) \n
        Initializes the Graph Convolutional Network model.

        @param n_hid: list of numbers of neurons on each layer of the network.
        @param activations: list of names of activation functions of each layer.
        @param lr: learning rate.
        """
        if n_hid is None:
            n_hid = np.array([self.__X.shape[1], self.__d])
        if activations is None:
            activations = np.repeat("sigmoid", len(n_hid))
            activations[-1] = "softmax"
        model_layers = [None] * len(n_hid)
        model_layers[0] = tf.keras.Input(shape=[n_hid[0]],
                                         batch_size=None,
                                         name="0")

        model_layers[1:] = [GCN.GCNLayer(output_dim=n_hid[i], activation_id=activations[i], A_matrix=self.__A_weighted)
                            for i in range(1, len(n_hid))]
        model = tf.keras.Sequential(model_layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.__optimizer = optimizer
        model.compile(optimizer=optimizer, metrics=[tf.keras.metrics.Accuracy()])
        self.__model = model

    def __get_loss(self, model):
        y_pred = model(self.__X)
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        return scce(self.__Y, y_pred, sample_weight=self.__Y_mask)

    def __get_gradients(self, model):
        with tf.GradientTape() as tape:
            tape.watch(model.variables)
            L = self.__get_loss(model)
        g = tape.gradient(L, model.variables)
        return g

    @Model._fit_in_init_model_fit
    @Model._verify_parameters(rules_dict=fit_verification)
    def fit(self,
            num_iter: int = 300,
            verbose: bool = True):
        """
        GCN - fit (step IV) \n
        Trains the GCN model.

        @param num_iter: Number of iterations
        @param verbose: Verbosity parameter
        """

        accuracy = tf.keras.metrics.Accuracy()
        for i in range(num_iter):
            g = self.__get_gradients(self.__model)
            self.__optimizer.apply_gradients(zip(g, self.__model.variables))
            if verbose:
                print("Epoch " + str(i+1) + ": " + str(self.__get_loss(self.__model)))
                accuracy.reset_states()
                accuracy.update_state(self.__Y, np.argmax(self.__model(self.__X), axis=1))
                print("Accuracy: " + str(accuracy.result().numpy()))

    @Model._embed_in_init_model_fit
    def embed(self):
        """
        GCN - embed (step V) \n
        Returns the embedding matrix.
        @return: The embedding matrix with shape N x d, where d is the number of unique labels.
        """
        return self.__model(self.__X)

    @staticmethod
    @Model._verify_parameters(rules_dict=fast_embed_verification)
    def fast_embed(graph: Graph,
                   Y: ndarray,
                   Y_mask: ndarray = None,
                   X: ndarray = None,
                   n_hid = None,
                   activations = None,
                   lr: float = 0.01,
                   num_iter: int = 300,
                   fit_verbose=True):
        """
        GCN - fast_embed \n
        Performs the embedding in a single step.

        @param graph: The graph to be embedded. Present in '__init__'
        @param Y: The vector of labels for vertices. Present in 'initialize'
        @param Y_mask: A binary vector masking the vertices from loss function calculation. Vertices corresponding to 1
        are included in loss function calculation, vertices corresponding to 0 are not. Present in 'initialize'
        @param X: Matrix of features of vertices. Identity matrix passed if X is None. Present in 'initialize'
        @param n_hid: list of numbers of neurons on each layer of the network. Present in 'initialize_model'
        @param activations: list of names of activation functions of each layer. Present in 'initialize_model'
        @param lr: learning rate. Present in 'initialize_model'
        @param num_iter: Number of iterations. Present in 'fit'
        @param fit_verbose: Verbosity parameter. Present in 'fit'
        @return: The embedding matrix with shape N x d, where d is the number of unique labels.
        """
        gcn = GCN(graph)
        gcn.initialize(Y, Y_mask, X)
        gcn.initialize_model(n_hid, activations, lr)
        gcn.fit(num_iter, verbose=fit_verbose)
        return gcn.embed()


