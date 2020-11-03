from engthesis.model import Model
from engthesis.metrics import metrics as met
from networkx import Graph, to_numpy_array
import tensorflow as tf
import numpy as np
from tensorflow_addons.losses import metric_learning


class SDNE(Model):

    def __init__(self,
                 graph: Graph):
        super().__init__(graph)
        self.__d = None
        self.__alpha = None
        self.__nu = None
        self.__A = None
        self.__A_tensor = None
        self.__N = None
        self.__beta = None
        self.__K = None
        self.__B = None
        self.__hid = None
        self.__model = None
        self.__optimizer = None

    @Model._init_in_init_model_fit
    def initialize(self,
                   alpha: float = 1,
                   beta: float = 0.3,
                   nu: float = 0.01):
        self.__alpha = alpha
        self.__beta = beta
        self.__nu = nu
        graph = self.get_graph()
        self.__A = to_numpy_array(graph, nodelist=np.arange(len(graph.nodes))).astype(np.float32)
        self.__A_tensor = tf.convert_to_tensor(self.__A)
        self.__N = len(graph.nodes)
        self.__B = tf.add(tf.multiply(self.__A, self.__beta), tf.ones([self.__N, self.__N]))

    @Model._init_model_in_init_model_fit
    def initialize_model(self,
                         d: int = 2,
                         n_layers: int = 2,
                         hidden_layers=None,
                         lr: float = 0.1):
        self.__K = n_layers
        self.__d = d
        if hidden_layers is None:
            self.__hid = [None] * n_layers
            self.__hid[0] = self.__N
            if n_layers > 2:
                self.__hid[1:] = np.arange(self.__d, self.__N, (self.__N - d) // (n_layers - 1))[::-1][(-n_layers + 1):]
            else:
                self.__hid[1] = self.__d
        else:
            assert (hidden_layers[0] == self.__N and hidden_layers[-1] == d)
            self.__hid = hidden_layers

        hid = self.__hid
        hid_all = [None] * (2 * self.__K - 1)
        hid_all[:self.__K] = hid
        hid_all[self.__K:] = list(reversed(hid))[1:]
        model_layers = [None] * len(hid_all)
        model_layers[0] = tf.keras.Input(shape=[self.__N],
                                         batch_size=None,
                                         name="0")
        activations = [tf.nn.sigmoid] * len(hid_all)
        activations[-1] = tf.keras.activations.linear
        model_layers[1:] = [tf.keras.layers.Dense(hid_all[i], activation=activations[i],
                                                  name=str(i)) for i in range(1, len(hid_all))]
        model = tf.keras.Sequential(model_layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.__optimizer = optimizer
        self.__model = model
        self.__model.compile(optimizer=optimizer)

    def get_hid(self):
        return self.__hid

    def info(self) -> str:
        raise NotImplementedError

    def get_nth_layer_output(self, model, n):
        partial = tf.keras.Model(model.inputs, model.layers[n].output)
        return partial(self.__A, training=False)

    def __loss1st(self, model):
        hidden_output = self.get_nth_layer_output(model, self.__K - 2)
        return tf.math.square(tf.norm(tf.multiply(self.__A_tensor, metric_learning.pairwise_distance(hidden_output))))

    def __loss2nd(self, x_hat):
        return tf.math.square(tf.norm(tf.multiply(self.__B, tf.add(x_hat, -self.__A))))

    @staticmethod
    def __lossReg(model):
        weights = [model.layers[i].get_weights()[0] for i in range(len(model.layers))]

        return tf.add_n([tf.math.square(tf.norm(tf.convert_to_tensor(i))) for i in weights])

    def __get_loss(self, model):
        output = model(self.__A)
        return self.__loss1st(model) + self.__alpha * self.__loss2nd(output) + self.__nu * self.__lossReg(model)

    def __get_gradients(self, model):
        with tf.GradientTape() as tape:
            tape.watch(model.variables)
            L = self.__get_loss(model)
        g = tape.gradient(L, model.variables)
        return g

    def get_decoded_matrix(self):
        return self.__model(self.__A).numpy()

    @Model._fit_in_init_model_fit
    def fit(self,
            num_iter=300,
            verbose=False
            ):
        for i in range(num_iter):
            g = self.__get_gradients(self.__model)
            self.__optimizer.apply_gradients(zip(g, self.__model.variables))
            if verbose:

                print("Epoch " + str(i + 1) + ": " + str(self.__get_loss(self.__model)))
                print("RMSE:" + str(met.rmse(self.__A, self.get_decoded_matrix())))

    @Model._embed_in_init_model_fit
    def embed(self) -> np.ndarray:
        return self.get_nth_layer_output(self.__model, self.__K - 2).numpy()

    @staticmethod
    def fast_embed(graph: Graph,
                   alpha: float = 1,
                   beta: float = 0.3,
                   nu: float = 0.01,
                   d: int = 2,
                   n_layers: int = 2,
                   hidden_layers = None,
                   lr: float = 0.1,
                   num_iter: int = 300,
                   fit_verbose: bool = False):
        sdne = SDNE(graph)
        sdne.initialize(alpha=alpha,
                        beta=beta,
                        nu=nu)
        sdne.initialize_model(d=d,
                              n_layers=n_layers,
                              hidden_layers=hidden_layers,
                              lr=lr)
        sdne.fit(num_iter=num_iter,
                 verbose=fit_verbose)
        return sdne.embed()
