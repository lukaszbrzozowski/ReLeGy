from engthesis.model import Model
from numpy import ndarray, arange
from networkx import Graph, to_numpy_array
import tensorflow as tf
import warnings


class GraphFactorization(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 lmbd: float = 0.1):
        A = to_numpy_array(graph, nodelist=arange(len(graph.nodes)))
        mask = (A > 0)
        self.__A = tf.constant(A, dtype="float32")
        self.__N = tf.constant(A.shape[0])
        self.__mask = tf.constant(mask, dtype="float32")
        self.__lmbd = tf.constant(lmbd)
        self.__d = tf.constant(d)
        self.__model = None
        super().__init__(graph)

    def info(self) -> str:
        raise NotImplementedError

    def __get_loss(self, model):
        y_pred = model(tf.eye(self.__N))
        main_loss = 0.5*tf.reduce_sum(tf.multiply(self.__mask,
                                                  tf.math.pow(self.__A - tf.matmul(y_pred, tf.transpose(y_pred)), 2)))
        reg_loss = self.__lmbd/2 * tf.pow(tf.norm(y_pred), 2)
        return main_loss+reg_loss

    def __get_gradients(self, model):
        with tf.GradientTape() as tape:
            tape.watch(model.variables)
            L = self.__get_loss(model)
        g = tape.gradient(L, model.variables)
        return g

    def get_loss(self):
        if self.__model is None:
            self.build_model()
            warnings.warn("The model has been built prior to calculating loss")
        return self.__get_loss(self.__model)

    def build_model(self,
                    optimizer: str = "adam",
                    lr: float = 0.1,
                    verbose: bool = False):

        input_layer = tf.keras.Input(shape=[self.__N],
                                     batch_size=None)
        output_layer = tf.keras.layers.Dense(self.__d, activation="linear")
        model = tf.keras.Sequential([input_layer, output_layer])
        optimizer_ent = tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": lr}})
        model.compile(optimizer=optimizer_ent)
        self.__model = model
        if verbose:
            print("The model has been built")

    def embed(self,
              num_epoch: int = 300,
              verbose: bool = False) -> ndarray:
        if self.__model is None:
            self.build_model()
            warnings.warn("The model has been built with default parameters")
        model = self.__model
        optimizer = model.optimizer
        for i in range(num_epoch):
            g = self.__get_gradients(model)
            optimizer.apply_gradients(zip(g, model.variables))
            if verbose:
                print("Epoch " + str(i+1) + ": " + str(self.__get_loss(model).numpy()))
        self.__model = model
        return self.__model(tf.eye(self.__N)).numpy()



