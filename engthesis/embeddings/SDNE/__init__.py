from engthesis.model import Model
from networkx import Graph, to_numpy_array
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow_addons.losses import metric_learning

class SDNE(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 alpha: float = 1,
                 beta: float = 0.3,
                 nu: float = 0.01,
                 n_layers: int = 2,
                 hidden_layers=None):
        super().__init__(graph)
        self.__d = d
        self.__alpha = alpha
        self.__nu = nu
        self.__A = to_numpy_array(graph).astype(np.float32)
        self.__A_tensor = tf.convert_to_tensor(self.__A)
        self.__N = len(graph.nodes)
        self.__beta = beta
        self.__K = n_layers
        self.__B = tf.add(tf.multiply(self.__A, self.__beta), tf.ones([self.__N, self.__N]))
        if hidden_layers is None:
            self.__hid = [None] * n_layers
            self.__hid[0] = self.__N
            if n_layers > 2:
                self.__hid[1:] = np.arange(d, self.__N, (self.__N-d)//(n_layers-1))[::-1][(-n_layers+1):]
            else:
                self.__hid[1] = self.__d
        else:
            assert(hidden_layers[0] == self.__N and hidden_layers[-1] == d)
            self.__hid = hidden_layers
        self.model = None

    def get_hid(self):
        return self.__hid

    def info(self) -> str:
        raise NotImplementedError

    def get_nth_layer_output(self, model, n):
        partial = tf.keras.Model(model.inputs, model.layers[n].output)
        return partial(self.__A, training=False)

    def loss1st(self, model):
        hidden_output = self.get_nth_layer_output(model, self.__K-2)
        return tf.math.square(tf.norm(tf.multiply(self.__A_tensor, metric_learning.pairwise_distance(hidden_output))))

    def loss2nd(self, x_hat):
        return tf.math.square(tf.norm(tf.multiply(self.__B, tf.add(x_hat, -self.__A))))

    def lossReg(self, model):
        weights = [model.layers[i].get_weights()[0] for i in range(len(model.layers))]

        return tf.add_n([tf.math.square(tf.norm(tf.convert_to_tensor(i))) for i in weights])

    def get_loss(self, model):
        output = model(self.__A)
        return self.loss1st(model) + self.__alpha*self.loss2nd(output) + self.__nu*self.lossReg(model)

    def get_gradients(self, model):
        with tf.GradientTape() as tape:
            tape.watch(model.variables)
            L = self.get_loss(model)
        g = tape.gradient(L, model.variables)
        return g

    def get_decoded_matrix(self):
        return self.model(self.__A).numpy()
    def embed(self,
              num_epoch=300,
              lr=0.01,
              verbose=False
              ):
        hid = self.__hid
        hid_all = [None] * (2*self.__K-1)
        hid_all[:self.__K] = hid
        hid_all[self.__K:] = list(reversed(hid))[1:]
        model_layers = [None] * len(hid_all)
        model_layers[0] = tf.keras.Input(shape=[self.__N],
                                         batch_size=None,
                                         name="0")
        activations = [tf.nn.sigmoid for i in range(len(hid_all))]
        activations[-1] = tf.keras.activations.linear
        model_layers[1:] = [tf.keras.layers.Dense(hid_all[i], activation=activations[i],
                                                  name=str(i)) for i in range(1, len(hid_all))]
        model = tf.keras.Sequential(model_layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer)
        for i in range(num_epoch):
            g = self.get_gradients(model)
            optimizer.apply_gradients(zip(g, model.variables))
            if verbose:
                print("Epoch " + str(i+1) + ": " + str(self.get_loss(model)))

        self.model = model
        return self.get_nth_layer_output(model, self.__K-2).numpy()


