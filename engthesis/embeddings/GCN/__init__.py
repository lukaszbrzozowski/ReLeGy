from engthesis.model import Model
import networkx as nx
from networkx import Graph
import numpy as np
from numpy import ndarray
import tensorflow as tf
import scipy.sparse as sps
from tensorflow.keras.layers import Layer

class GCN(Model):

    def __init__(self,
                 graph: Graph,
                 Y: ndarray,
                 X: ndarray = None,
                 d: int = 2
                 ):
        super().__init__(graph)
        self.__d = d
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

        A_hat = self.__A + sps.identity(self.__A.shape[0])
        D = sps.diags(A_hat.sum(axis=0).A1)
        D_sqrt_inv = D.sqrt().power(-1)
        result = D_sqrt_inv @ A_hat @ D_sqrt_inv

        self.__A_weighted = tf.convert_to_tensor(result.toarray().astype(np.float32))
        self.__Y = Y
        self.model = None

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

    def get_loss(self, model):
        y_pred = model(self.__X)
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        return scce(self.__Y, y_pred)

    def get_gradients(self, model):
        with tf.GradientTape() as tape:
            tape.watch(model.variables)
            L = self.get_loss(model)
        g = tape.gradient(L, model.variables)
        return g

    def info(self) -> str:
        raise NotImplementedError

    def embed(self, n_hid=None,
              num_epoch=300,
              activations=None,
              lr=0.01,
              verbose=True) -> ndarray:
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
        model.compile(optimizer=optimizer, metrics=[tf.keras.metrics.Accuracy()])
        accuracy = tf.keras.metrics.Accuracy()
        for i in range(num_epoch):
            g = self.get_gradients(model)
            optimizer.apply_gradients(zip(g, model.variables))
            if verbose:
                print("Epoch " + str(i+1) + ": " + str(self.get_loss(model)))
                accuracy.reset_states()
                accuracy.update_state(self.__Y, np.argmax(model(self.__X), axis=1))
                print("Accuracy: " + str(accuracy.result().numpy()))

        self.model = model

        return self.model(self.__X)

