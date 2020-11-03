from engthesis.model import Model
from networkx import to_numpy_array, Graph
from numpy import ndarray, arange, empty
import tensorflow as tf
import tensorflow_probability as tfp
from gensim.models import word2vec


class DeepWalk(Model):

    def __init__(self,
                 graph: Graph):

        super().__init__(graph)
        self.__model = None
        self.__N = None
        self.__A = None
        self.__T = None
        self.__gamma = None
        self.__rw = None
        self.__d = None

    @Model._init_in_init_model_fit
    def initialize(self,
                   T: int = 40,
                   gamma: int = 1):
        graph = self.get_graph()
        self.__N = len(graph.nodes)
        self.__A = tf.constant(to_numpy_array(graph, nodelist=arange(self.__N)), dtype="float32")
        self.__T = tf.constant(T)
        self.__gamma = tf.constant(gamma)
        self.__rw = self.__generate_random_walks()

    @Model._init_model_in_init_model_fit
    def initialize_model(self,
                         d: int = 2,
                         alpha: float = 0.025,
                         min_alpha: float = 0.0001,
                         window: int = 5,
                         hs: int = 1,
                         negative: int = 0):

        model = word2vec.Word2Vec(sentences=None,
                                  size=d,
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
        self.__model.build_vocab(sentences=self.__rw)
        self.__d = d

    def __generate_random_walks(self):
        D = tf.linalg.diag(tf.pow(tf.reduce_sum(self.__A, 1), -1))
        P = tfp.distributions.Categorical(probs=tf.tile(tf.matmul(D, self.__A), [self.__gamma, 1]))
        temp_dist = tfp.distributions.Categorical(probs=tf.tile(tf.eye(self.__N), [self.__gamma, 1]))
        hmm = tfp.distributions.HiddenMarkovModel(temp_dist,
                                                  P,
                                                  temp_dist,
                                                  self.__T)
        return hmm.sample().numpy().astype("str").tolist()

    def get_random_walks(self):
        return self.__rw

    def info(self) -> str:
        raise NotImplementedError

    @Model._fit_in_init_model_fit
    def fit(self,
            num_iter=300):

        self.__model.train(self.__rw,
                           epochs=num_iter,
                           total_examples=len(self.__rw))

    @Model._embed_in_init_model_fit
    def embed(self) -> ndarray:
        ret_matrix = empty((self.__N, self.__d), dtype="float32")
        for i in arange(self.__N):
            ret_matrix[i, :] = self.__model.wv[str(i)]
        return ret_matrix

    @staticmethod
    def fast_embed(graph: Graph,
                   T: int = 40,
                   gamma: int = 1,
                   d: int = 2,
                   alpha: float = 0.025,
                   min_alpha: float = 0.0001,
                   window: int = 5,
                   hs: int = 1,
                   negative: int = 0,
                   num_iter=300) -> ndarray:
        dw = DeepWalk(graph)
        dw.initialize(T=T,
                      gamma=gamma)
        dw.initialize_model(d=d,
                            alpha=alpha,
                            min_alpha=min_alpha,
                            window=window,
                            hs=hs,
                            negative=negative)
        dw.fit(num_iter=num_iter)
        return dw.embed()


