from gensim.models import word2vec
import networkx as nx
from networkx import Graph
import numpy as np
from numpy import ndarray

from relegy.__base import Model

construct_verification = {"graph": [(lambda x: issubclass(type(x), Graph), "'graph' must be a networkx graph")]}

init_verification = {"T" : [(lambda x: x > 0, "T must be greater than 0.")],
                     "gamma" : [(lambda x: x > 0, "gamma must be greater than 0.")],
                     "p": [(lambda x: x > 0, "'p' must be greater than 0.")],
                     "q": [(lambda x: x > 0, "'q' must be greater than 0.")]
                     }

init_model_verification = {"d": [(lambda x: x > 0, "d must be greater than 0.")],
                           "alpha": [(lambda x: x > 0, "alpha must be greater than 0.")],
                           "min_alpha": [(lambda x: x > 0, "min_alpha must be greater than 0.")],
                           "window": [(lambda x: x > 0, "window must be greater than 0.")],
                           "hs": [(lambda x: 0 <= x <= 1, "hs must be boolean or either 0 or 1")],
                           "negative": [(lambda x: x >= 0, "negative must be non-negative")]}

fit_verification = {"num_iter": [(lambda x: x > 0, "num_iter must be greater than 0")]}

fast_embed_verification = Model.dict_union(construct_verification, init_verification, init_model_verification, fit_verification)


class Node2Vec(Model):
    """
    The Node2Vec method implementation. \n
    The details may be found in: \n
    'A. Grover and J. Leskovec. node2vec: Scalable feature learning for networks. In KDD, 2016'
    """

    @Model._verify_parameters(rules_dict=construct_verification)
    def __init__(self,
                 graph: Graph) -> None:
        """
        Node2Vec - constructor (step I)

        @param graph: The graph to be embedded. Nodes of the graph must be a sorted array from 0 to n-1, where n is
        the number of vertices.
        """

        super().__init__(graph)
        self.__N = None
        self.__A = None
        self.__T = None
        self.__gamma = None
        self.__p = None
        self.__q = None
        self.__rw = None
        self.__model = None
        self.__d = None

    @Model._init_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_verification)
    def initialize(self,
                   T: int = 40,
                   gamma: int = 1,
                   p: float = 1,
                   q: float = 1):
        """
        Node2Vec - initialize (step II) \n
        Calculates the random walks on the graph.

        @param T: The length of a single random walk.
        @param gamma: Number of random walks starting at a single vertex.
        @param p: Bias parameter of the random walks, as described in the article.
        @param q: Bias parameter of the random walks, as described in the article.
        """
        graph = self.get_graph()
        self.__N = len(graph.nodes)
        self.__A = nx.to_numpy_array(graph, nodelist=np.arange(self.__N))
        self.__T = T
        self.__gamma = gamma
        self.__p = p
        self.__q = q
        self.__rw = self.__generate_random_walks()

    def __generate_random_walks(self):
        """
        Returns the random walks on the graph.
        """
        random_walks: ndarray = np.empty((self.__N * self.__gamma, self.__T))
        for i in range(self.__gamma):
            # random_walk_matrix contains random walks for 1 iteration of i
            random_walk_matrix = np.empty((self.__N, self.__T))
            vertices = np.arange(self.__N)
            random_walk_matrix[:, 0] = vertices
            probabilities = np.multiply(self.__A, 1 / np.repeat(np.sum(self.__A, axis=1), [self.__N])
                                        .reshape(self.__N, self.__N))
            # generation 2nd step of the random walks with uniform probability
            next_vertices = [np.random.choice(vertices, size=1, p=np.asarray(probabilities[it, :]).reshape(-1))[0]
                             for it in range(self.__N)]
            random_walk_matrix[:, 1] = next_vertices
            for j in range(2, self.__T):
                probabilities_mask = np.ones((self.__N, self.__N))
                # generating mask on previous vertices in the random walks
                mask_p = np.identity(self.__N)[vertices, :]
                # generating mask on the vertices which are approachable from current vertices, unapproachable from
                # previous vertices and are not the previous vertices
                mask_q = np.logical_and(self.__A[next_vertices, :] > 0,
                                        np.logical_and(np.logical_and(self.__A @ self.__A > 0, self.__A <= 0),
                                                       np.logical_not(np.identity(self.__N)))[vertices, :])
                # modifying probabilities' mask accordingly
                probabilities_mask = np.where(mask_p, probabilities_mask / self.__p, probabilities_mask)
                probabilities_mask = np.where(mask_q, probabilities_mask / self.__q, probabilities_mask)
                probabilities = np.multiply(self.__A[next_vertices, :], probabilities_mask)
                # normalizing probabilities
                normalized_probabilities = np.multiply(probabilities, 1 / np.repeat(np.sum(probabilities, axis=1),
                                                                                    [self.__N])
                                                       .reshape(self.__N, self.__N))
                cur_next_vertices = [np.random.choice(np.arange(self.__N), size=1,
                                                      p=np.asarray(normalized_probabilities[it, :]).reshape(-1))[0]
                                     for it in range(self.__N)]
                random_walk_matrix[:, j] = cur_next_vertices
                vertices, next_vertices = next_vertices, cur_next_vertices

            random_walks[(i * self.__N):((i + 1) * self.__N), :] = random_walk_matrix

        return random_walks.astype(int).astype(str).tolist()

    def get_random_walks(self):
        """
        Returns the random walks generated on the graph. Can be used only after the 'initialize' step.
        """
        if not self._initialized:
            raise Exception("Cannot be used before the 'initialize' step")
        return self.__rw

    @Model._init_model_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_model_verification)
    def initialize_model(self,
                         d: int = 2,
                         alpha=0.025,
                         min_alpha=0.0001,
                         window=5,
                         hs=0,
                         negative=5):
        """
        Node2Vec - initialize_model (step III) \n
        Generates the Word2Vec network model.
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

    @Model._fit_in_init_model_fit
    @Model._verify_parameters(rules_dict=fit_verification)
    def fit(self,
            num_iter=300):
        """
        Node2Vec - fit (step IV) \n
        Trains the Word2Vec SkipGram network on the vocabulary of the previously generated random walks.
        @param num_iter: Number of iterations of the training.
        """
        self.__model.train(self.__rw,
                           epochs=num_iter,
                           total_examples=len(self.__rw))

    @Model._embed_in_init_model_fit
    def embed(self):
        """
        Node2Vec - embed (step V) \n
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
                   p: float = 1,
                   q: float = 1,
                   d: int = 2,
                   alpha: float = 0.025,
                   min_alpha: float = 0.0001,
                   window: int = 5,
                   hs: int = 0,
                   negative: int = 5,
                   num_iter: int = 300):
        """
        Node2Vec - fast_embed \n
        Returns the embedding in a single step.

        @param graph: The graph to be embedded. Present in '__init__'
        @param T: The length of a single random walk. Present in 'initialize'
        @param gamma: Number of random walks starting at a single vertex. Present in 'initialize'
        @param d: The embedding dimension. Present in 'initialize_model'
        @param p: Bias parameter of the random walks, as described in the article. Present in 'initialize'
        @param q: Bias parameter of the random walks, as described in the article. Present in 'initialize'
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
        n2v = Node2Vec(graph)
        n2v.initialize(T=T,
                       gamma=gamma,
                       p=p,
                       q=q)
        n2v.initialize_model(d=d,
                             alpha=alpha,
                             min_alpha=min_alpha,
                             window=window,
                             hs=hs,
                             negative=negative)
        n2v.fit(num_iter=num_iter)
        return n2v.embed()
