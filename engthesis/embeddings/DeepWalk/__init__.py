from engthesis.model import Model
from networkx import to_numpy_array, Graph
from numpy import ndarray, arange, empty
import tensorflow as tf
import tensorflow_probability as tfp
from gensim.models import word2vec
import warnings

class DeepWalk(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 T: int = 40,
                 gamma: int = 1):

        super().__init__(graph)
        self.__model = None
        self.__N = len(graph.nodes)
        self.__A = tf.constant(to_numpy_array(graph, nodelist=arange(self.__N)), dtype="float32")
        self.__d = d
        self.__T = tf.constant(T)
        self.__gamma = tf.constant(gamma)

    def generate_random_walks(self):
        D = tf.linalg.diag(tf.pow(tf.reduce_sum(self.__A, 1), -1))
        P = tfp.distributions.Categorical(probs=tf.tile(tf.matmul(D, self.__A), [self.__gamma, 1]))
        temp_dist = tfp.distributions.Categorical(probs=tf.tile(tf.eye(self.__N), [self.__gamma, 1]))
        hmm = tfp.distributions.HiddenMarkovModel(temp_dist,
                                                  P,
                                                  temp_dist,
                                                  self.__T)
        return hmm.sample()

    def info(self) -> str:
        raise NotImplementedError

    def build_model(self,
                    alpha: float = 0.025,
                    min_alpha: float = 0.0001,
                    window: int = 5,
                    hs: int = 1,
                    negative: int = 0):
        model = word2vec.Word2Vec(sentences=None,
                                  size=self.__d,
                                  min_count=1,
                                  negative=negative,
                                  alpha=alpha,
                                  min_alpha=min_alpha,
                                  sg=1,
                                  hs=hs,
                                  window=window,
                                  sample=0,
                                  sorted_vocab=0)
        self.__model = model

    def embed(self,
              num_epoch=300) -> ndarray:
        if self.__model is None:
            warnings.warn("The model has been with default parameters prior to embedding")
            self.build_model()
        rw = self.generate_random_walks().numpy().astype(str).tolist()
        self.__model.build_vocab(sentences=rw)
        self.__model.train(rw,
                           epochs=num_epoch,
                           total_examples=len(rw))
        ret_matrix = empty((self.__N, self.__d), dtype="float32")
        for i in arange(self.__N):
            ret_matrix[i, :] = self.__model.wv[str(i)]
        return ret_matrix

