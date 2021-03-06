from relegy.__base import Model
import networkx as nx
from networkx import Graph
import numpy as np
from numpy import ndarray
import tensorflow as tf
import tensorflow_probability as tfp
from gensim.models import word2vec

construct_verification = {"graph": [(lambda x: type(x) == Graph, "'graph' must be a networkx Graph")]}

init_verification = {"T": [(lambda x: x > 0, "'T' must be greater than 0.")],
                     "gamma": [(lambda x: x > 0, "'gamma' must be greater than 0.")]}

init_model_verification = {"d": [(lambda x: x > 0, "'d' must be greater than 0.")],
                           "alpha": [(lambda x: x > 0, "'alpha' must be greater than 0.")],
                           "min_alpha": [(lambda x: x > 0, "'min_alpha' must be greater than 0.")],
                           "window": [(lambda x: x > 0, "'window' must be greater than 0.")],
                           "hs": [(lambda x: 0 <= x <= 1, "'hs' must be boolean or either 0 or 1")],
                           "negative": [(lambda x: x >= 0, "'negative' must be non-negative")]}

fit_verification = {"num_iter": [(lambda x: x > 0, "'num_iter' must be greater than 0")]}

fast_embed_verification = Model.dict_union(construct_verification, init_verification, init_model_verification, fit_verification)


class DeepWalk(Model):
    """
    The DeepWalk method implementation. \n
    The details may be found in: \n
    'B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning of social representations. In KDD, 2014'
    """

    @Model._verify_parameters(rules_dict=construct_verification)
    def __init__(self,
                 graph: Graph):
        """
        DeepWalk - constructor (step I)

        @param graph: The graph to be embedded. Nodes of the graph must be a sorted array from 0 to n-1, where n is
        the number of vertices. May be weighted, but cannot be directed.
        """

        super().__init__(graph)
        self.__model = None
        self.__N = None
        self.__A = None
        self.__T = None
        self.__gamma = None
        self.__rw = None
        self.__d = None

    @Model._init_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_verification)
    def initialize(self,
                   T: int = 40,
                   gamma: int = 1):
        """
        DeepWalk - initialize (step II) \n
        Calculates the random walks on the graph.

        @param T: The length of a single random walk.
        @param gamma: Number of random walks starting at a single vertex.
        """
        graph = self.get_graph()
        self.__N = len(graph.nodes)
        self.__A = tf.constant(nx.to_numpy_array(graph, nodelist=np.arange(self.__N)), dtype="float32")
        self.__T = tf.constant(T)
        self.__gamma = tf.constant(gamma)
        self.__rw = self.__generate_random_walks()

    @Model._init_model_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_model_verification)
    def initialize_model(self,
                         d: int = 2,
                         alpha: float = 0.025,
                         min_alpha: float = 0.0001,
                         window: int = 5,
                         hs: int = 1,
                         negative: int = 0):
        """
        DeepWalk - initialize_model (step III) \n
        Initializes the Word2Vec network model.
        @param d: The embedding dimension.
        @param alpha: The starting learning rate of the model, as described in gensim.Word2Vec documentation.
        @param min_alpha: The minimal learning rate of the model, as described in gensim.Word2Vec documentation.
        @param window: The window size of the network, as described in gensim.Word2Vec documentation.
        @param hs: Must be 0 or 1. If 1, Hierarchical Softmax is used, as described in gensim.Word2Vec documentation.
        @param negative: Number of samples for the Negative Sampling method, as described in gensim.Word2Vec
        documentation. If 0, the Negative Sampling is not used.
        """

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
        """
        Generates the random walks on the graph.
        """
        D = tf.linalg.diag(tf.pow(tf.reduce_sum(self.__A, 1), -1))
        P = tfp.distributions.Categorical(probs=tf.tile(tf.matmul(D, self.__A), [self.__gamma, 1]))
        temp_dist = tfp.distributions.Categorical(probs=tf.tile(tf.eye(self.__N), [self.__gamma, 1]))
        hmm = tfp.distributions.HiddenMarkovModel(temp_dist,
                                                  P,
                                                  temp_dist,
                                                  self.__T)
        return hmm.sample().numpy().astype("str").tolist()

    def get_random_walks(self):
        """
        Returns the random walks generated on the graph. Can be used only after the 'initialize' step.
        """
        if not self._initialized:
            raise Exception("Cannot be used before the 'initialize' step")
        return self.__rw

    @Model._fit_in_init_model_fit
    @Model._verify_parameters(rules_dict=fit_verification)
    def fit(self,
            num_iter=300):
        """
        DeepWalk - fit (step IV) \n
        Trains the Word2Vec SkipGram network on the vocabulary of the previously generated random walks.
        @param num_iter: Number of iterations of the training.
        """
        self.__model.train(self.__rw,
                           epochs=num_iter,
                           total_examples=len(self.__rw))

    @Model._embed_in_init_model_fit
    def embed(self) -> ndarray:
        """
        DeepWalk - embed (step V) \n
        Returns the embedding from the Word2Vec network.
        @return: The embedding matrix.
        """
        ret_matrix = np.empty((self.__N, self.__d), dtype="float32")
        for i in np.arange(self.__N):
            ret_matrix[i, :] = self.__model.wv[str(i)]
        return ret_matrix

    @staticmethod
    @Model._verify_parameters(rules_dict=fast_embed_verification)
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
        """
        DeepWalk - fast_embed \n
        Returns the embedding in a single step.

        @param graph: The graph to be embedded. Present in '__init__'
        @param T: The length of a single random walk. Present in 'initialize'
        @param gamma: Number of random walks starting at a single vertex. Present in 'initialize'
        @param d: The embedding dimension. Present in 'initialize_model'
        @param alpha: The starting learning rate of the model, as described in gensim.Word2Vec documentation. Present
        in 'initialize_model'
        @param min_alpha: The minimal learning rate of the model, as described in gensim.Word2Vec documentation. Present
        in 'initialize_model'
        @param window: The window size of the network, as described in gensim.Word2Vec documentation. Present in
        'initialize_model'
        @param hs: Must be 0 or 1. If 1, Hierarchical Softmax is used, as described in gensim.Word2Vec documentation.
        Present in 'initialize_model'
        @param negative: Number of samples for the Negative Sampling method, as described in gensim.Word2Vec
        documentation. If 0, the Negative Sampling is not used. Present in 'initialize_model'
        @param num_iter: Number of iterations of the training. Present in 'fit'
        @return: The embedding matrix.
        """
        dw = DeepWalk(graph=graph)
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
