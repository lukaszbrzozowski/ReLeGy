import numpy as np
from fastdtw import fastdtw
from gensim.models import Word2Vec
from networkx import Graph
import networkx as nx
from numpy import ndarray
from six import iteritems

from relegy.__base import Model

init_verification = {"T" : [(lambda x: x > 0, "'T' must be greater than 0.")],
                     "gamma" : [(lambda x: x > 0, "'gamma' must be greater than 0.")],
                     "q": [(lambda x: x >= 0, "'q' must be non-negative")],
                     "OPT3_k": [(lambda x: True if x is None else x > 0, "'OPT3_k' must be greater than 0.")]}

init_model_verification = {"d": [(lambda x: x > 0, "'d' must be greater than 0.")],
                           "alpha": [(lambda x: x > 0, "'alpha' must be greater than 0.")],
                           "min_alpha": [(lambda x: x > 0, "'min_alpha' must be greater than 0.")],
                           "window": [(lambda x: x > 0, "'window' must be greater than 0.")],
                           "hs": [(lambda x: 0 <= x <= 1, "'hs' must be boolean or either 0 or 1")],
                           "negative": [(lambda x: x >= 0, "'negative' must be non-negative")]}

fit_verification = {"num_iter": [(lambda x: x > 0, "'num_iter' must be greater than 0")]}

fast_embed_verification = Model.dict_union(init_verification, init_model_verification, fit_verification)


class Struc2Vec(Model):
    """
    The Struc2Vec method implementation. \n
    The details may be found in: \n
    'L.F.R. Ribeiro, P.H.P. Saverese, and D.R. Figueiredo. struc2vec: Learning node representations from structural
identity. In KDD, 2017.'
    """

    def __init__(self,
                 graph: Graph):
        """
        Struc2Vec - constructor (step I)

        @param graph: Graph to be embedded.
        """

        super().__init__(graph)
        self.__d = None
        self.__T = None
        self.__gamma = None
        self.__q = None
        self.__k = None
        self.__model = None
        self.__OPT1 = None
        self.__N = None
        self.__rw = None

    @Model._init_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_verification)
    def initialize(self,
                   T: int = 40,
                   gamma: int = 1,
                   q: float = 0.3,
                   OPT1: bool = False,
                   OPT3_k: int = None):
        """
        Struc2Vec - initialize (step II) \n
        Generates the multigraph.

        @param T: length of a single random walk.
        @param gamma: Number of random walks starting from a single vertex.
        @param q: Probability of changing layers in the multigraph.
        @param OPT1: If True, OPT1 optimization is used, as described in the article.
        @param OPT3_k: If a number k is given instead of None, OPT3 optimization is used with parameter k, as described
        in the article.
        """
        graph = self.get_graph()
        self.__T = T
        self.__gamma = gamma
        self.__q = q
        self.__k = nx.diameter(graph)
        self.__N = len(graph.nodes)
        if OPT3_k is not None:
            assert (OPT3_k <= self.__k)
            self.__k = OPT3_k
        self.__OPT1 = OPT1

        matrix_dict = self.__generate_similarity_matrices()
        w_in, w_f = self.__generate_multigraph_edges(matrix_dict)
        self.__rw = self.__generate_random_walks(w_in, w_f)

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
        Struc2Vec - initialize_model (step III) \n
        Initializes the Word2Vec network model.

        @param d: The embedding dimension.
        @param alpha: The starting learning rate of the model, as described in gensim.Word2Vec documentation.
        @param min_alpha: The minimal learning rate of the model, as described in gensim.Word2Vec documentation.
        @param window: The window size of the network, as described in gensim.Word2Vec documentation.
        @param hs: Must be 0 or 1. If 1, Hierarchical Softmax is used, as described in gensim.Word2Vec documentation.
        @param negative: Number of samples for the Negative Sampling method, as described in gensim.Word2Vec
        documentation. If 0, the Negative Sampling is not used.
        """
        model = Word2Vec(alpha=alpha,
                         size=d,
                         min_alpha=min_alpha,
                         window=window,
                         sg=1,
                         hs=hs,
                         negative=negative,
                         min_count=1)
        self.__model = model
        self.__model.build_vocab(sentences=self.__rw)
        self.__d = d

    def __generate_similarity_matrices(self):
        deg_seq = np.array(self.get_graph().degree(np.arange(self.__N)))[:, 1].reshape(self.__N, -1)
        k_max = self.__k
        dist_matrix = nx.floyd_warshall_numpy(self.get_graph(), nodelist=np.arange(self.__N))
        f_cur = np.zeros((self.__N, self.__N))
        matrix_dict = {}
        for k in np.arange(k_max + 1):
            for u in np.arange(self.__N):
                mask_u = (dist_matrix[u, :] == k).reshape(self.__N, -1)
                if np.any(mask_u):
                    for v in np.arange(self.__N):
                        mask_v = (dist_matrix[v, :] == k).reshape(self.__N, -1)
                        if np.any(mask_v):
                            deg_u = deg_seq[mask_u]
                            deg_v = deg_seq[mask_v]
                            if self.__OPT1:
                                dist = self.__compare_deg_seq_opt1(deg_u, deg_v)
                            else:
                                dist = self.__compare_deg_seq(deg_u, deg_v)
                            f_cur[u, v] += dist
                        else:
                            f_cur[u, v] = None
                else:
                    f_cur[u, :] = None
            matrix_dict["F" + str(k)] = np.copy(f_cur)
        return matrix_dict

    @staticmethod
    def dist_fun(a, b):
        return (max(a[0], b[0]) / min(a[0], b[0]) - 1) * max(a[1], b[1])

    def __compare_deg_seq_opt1(self, deg_u, deg_v):
        uq1, cn1 = np.unique(deg_u, return_counts=True)
        uq2, cn2 = np.unique(deg_v, return_counts=True)
        a1 = dict(zip(uq1, cn1))
        a2 = dict(zip(uq2, cn2))
        counts_u = np.zeros((max(a1.keys()), 2))
        counts_v = np.zeros((max(a2.keys()), 2))
        counts_u[:, 0] = np.arange(1, counts_u.shape[0] + 1)
        counts_v[:, 0] = np.arange(1, counts_v.shape[0] + 1)
        for i in a1.keys():
            counts_u[i - 1, 1] = a1[i]
        for i in a2.keys():
            counts_v[i - 1, 1] = a2[i]
        dist, _ = fastdtw(counts_u, counts_v, dist=lambda a, b: self.dist_fun(a, b))
        return dist

    @staticmethod
    def __generate_multigraph_edges(matrix_dict):
        n_layers = len(matrix_dict)
        weights_in_layers = {"W" + str(i): np.nan_to_num(np.exp(-matrix_dict["F" + str(i)])) for i in range(n_layers)}
        avg_weights = [np.mean(weights_in_layers["W" + str(i)]) for i in range(n_layers)]
        gammas = [np.mean(weights_in_layers["W" + str(i)] > avg_weights[i], axis=1) for i in range(n_layers)]
        weights_forward = [np.log(gammas[i] + np.e) for i in range(n_layers - 1)]
        return weights_in_layers, weights_forward

    @staticmethod
    def __compare_deg_seq(deg1, deg2):
        dist_, _ = fastdtw(np.sort(deg1), np.sort(deg2), dist=lambda a, b: max(a, b) / min(a, b) - 1)
        return dist_

    @staticmethod
    def __generate_normalization(w_in):
        no_diag = [w_in["W" + str(i)] - np.diag(np.diag(w_in["W" + str(i)])) for i in range(len(w_in))]
        row_sums = [np.repeat(np.sum(w, axis=1).reshape(-1, 1), w.shape[0], axis=1) for w in no_diag]
        return np.array(no_diag) / np.array(row_sums)

    def __generate_random_walks(self, w_in, w_f):
        Z = self.__generate_normalization(w_in)
        N = len(self.get_graph().nodes)
        random_walks = np.empty((self.__gamma * N, self.__T))
        for i in range(self.__gamma):
            for j in range(N):
                random_walks[(i * N) + j, 0] = j
                rw_length = 1
                cur_layer = 0
                cur_v = j
                while rw_length < self.__T:
                    if np.random.random() > self.__q:
                        if cur_layer == 0:
                            cur_layer = 1
                        elif cur_layer == self.__k:
                            cur_layer = self.__k - 1
                        else:
                            p = w_f[cur_layer][cur_v] / (w_f[cur_layer][cur_v] + 1)
                            change_lay = True
                            if np.random.random() < p:
                                cur_layer += 1
                                if np.all(w_in["W" + str(cur_layer)][cur_v, :] == 0):
                                    cur_layer -= 1
                                    change_lay = False
                            else:
                                if change_lay:
                                    cur_layer -= 1
                    cur_w = Z[cur_layer]
                    next_v = np.random.choice(np.arange(N), p=cur_w[cur_v, :])
                    random_walks[(i * N) + j, rw_length] = next_v
                    cur_v = next_v
                    rw_length += 1
        return random_walks.astype(int).astype(str).tolist()

    def __get_weights_from_model(self):
        wv = self.__model.wv
        weight_matrix = np.empty((len(wv.vocab.keys()), self.__d + 1))
        i = 0
        temp_vocab = {int(k): v for k, v in wv.vocab.items()}
        for word, vocab in sorted(iteritems(temp_vocab)):
            row = wv.syn0[vocab.index]
            weight_matrix[i, 0] = word
            weight_matrix[i, 1:] = row
            i += 1
        return weight_matrix

    @Model._fit_in_init_model_fit
    @Model._verify_parameters(rules_dict=fit_verification)
    def fit(self, num_iter=1000):
        """
        Struc2Vec - fit (step IV) \n
        Trains the Word2Vec skipgram model.

        @param num_iter: Number of iterations.
        """
        self.__model.train(sentences=self.__rw,
                           total_examples=len(self.__rw),
                           epochs=num_iter)

    @Model._embed_in_init_model_fit
    def embed(self) -> ndarray:
        """
        Struc2Vec - embed (step V) \n
        Returns the embedding matrix from the Word2Vec network.
        @return: The embedding matrix.
        """
        return self.__get_weights_from_model()

    @staticmethod
    @Model._verify_parameters(rules_dict=fast_embed_verification)
    def fast_embed(graph: Graph,
                   T: int = 40,
                   gamma: int = 1,
                   q: float = 0.3,
                   OPT1: bool = False,
                   OPT3_k: int = None,
                   d: int = 2,
                   alpha: float = 0.025,
                   min_alpha: float = 0.0001,
                   window: int = 5,
                   hs: int = 1,
                   negative: int = 0,
                   num_iter=1000):
        """
        Struc2Vec - fast_embed \n
        Performs the embedding in a single step.

        @param graph: Graph to be embedded. Present in '__init__'
        @param T: length of a single random walk. Present in 'initialize'
        @param gamma: Number of random walks starting from a single vertex. Present in 'initialize'
        @param q: Probability of changing layers in the multigraph. Present in 'initialize'
        @param OPT1: If True, OPT1 optimization is used, as described in the article. Present in 'initialize'
        @param OPT3_k: If a number k is given instead of None, OPT3 optimization is used with parameter k, as described
        in the article. Present in 'initialize'
        @param d: The embedding dimension. Present in 'initialize_model'
        @param alpha: The starting learning rate of the model, as described in gensim.Word2Vec documentation.
        Present in 'initialize_model'
        @param min_alpha: The minimal learning rate of the model, as described in gensim.Word2Vec documentation. Present
        in 'initialize_model'
        @param window: The window size of the network, as described in gensim.Word2Vec documentation. Present in
        'initialize_model'
        @param hs: Must be 0 or 1. If 1, Hierarchical Softmax is used, as described in gensim.Word2Vec documentation.
        Present in 'initialize_model'
        @param negative: Number of samples for the Negative Sampling method, as described in gensim.Word2Vec
        documentation. If 0, the Negative Sampling is not used. Present in 'initialize_model'
        @param num_iter: Number of iterations. Present in 'fit'
        @return: The embedding matrix.
        """
        s2v = Struc2Vec(graph)
        s2v.initialize(T=T,
                       gamma=gamma,
                       q=q,
                       OPT1=OPT1,
                       OPT3_k=OPT3_k)
        s2v.initialize_model(d=d,
                             alpha=alpha,
                             min_alpha=min_alpha,
                             window=window,
                             hs=hs,
                             negative=negative)
        s2v.fit(num_iter)
        return s2v.embed()
