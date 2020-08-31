import numpy as np
from engthesis.model.base import Model
from networkx import Graph, diameter, floyd_warshall_numpy
from numpy import ndarray
from fastdtw import fastdtw
from gensim.models import Word2Vec

class Struct2Vec(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 q: float = 0.3,
                 T: int = 40,
                 gamma: int = 1,
                 window_size: int = 5):
        """

        :param graph:
        :param d:
        :param T:
        :param gamma:
        :param window_size:
        """
        super().__init__(graph)
        self.__d = d
        self.__T = T
        self.__gamma = gamma
        self.__window = window_size
        self.__q = q
        self.__k = diameter(self.get_graph())
        self.__model = None

    def generate_similarity_matrices(self):
        N = len(self.get_graph().nodes)
        deg_seq = np.array(self.get_graph().degree(np.arange(N)))[:, 1].reshape(N, -1)
        k_max = self.__k
        dist_matrix = floyd_warshall_numpy(self.get_graph())
        f_cur = np.zeros((N, N))
        matrix_dict = {}
        for k in np.arange(k_max+1):
            for u in np.arange(N):
                for v in np.arange(N):
                    mask_u = (dist_matrix[u, :] == k).reshape(N, -1)
                    mask_v = (dist_matrix[v, :] == k).reshape(N, -1)
                    if np.any(mask_u) and np.any(mask_v):
                        deg_u = deg_seq[mask_u]
                        deg_v = deg_seq[mask_v]
                        dist = self.__compare_deg_seq(deg_u, deg_v)
                        f_cur[u, v] += dist
                    else:
                        f_cur[u, v] = None
            matrix_dict["F" + str(k)] = np.copy(f_cur)
        return matrix_dict

    @staticmethod
    def generate_multigraph_edges(matrix_dict):
        n_layers = len(matrix_dict)
        weights_in_layers_with_nan = {"W" + str(i): np.exp(-matrix_dict["F"+str(i)]) for i in range(n_layers)}
        avg_weights = [np.nanmean(weights_in_layers_with_nan["W" + str(i)]) for i in range(n_layers)]
        masks = [np.logical_not(np.isnan(weights_in_layers_with_nan["W"+str(i)])) for i in range(n_layers)]
        gammas = [np.mean(weights_in_layers_with_nan["W" + str(i)][masks[i]] > avg_weights[i]) for i in range(n_layers)]
        weights_in_layers = {"W" + str(i): np.nan_to_num(weights_in_layers_with_nan["W" + str(i)]) for i in range(n_layers)}
        weights_forward = [np.log(gammas[i]+np.e) for i in range(n_layers-1)]
        return weights_in_layers, weights_forward

    @staticmethod
    def __compare_deg_seq(deg1, deg2):
        dist_, _ = fastdtw(np.sort(deg1), np.sort(deg2), dist=lambda a, b: max(a, b)/min(a, b) - 1)
        return dist_

    @staticmethod
    def __generate_normalization(w_in):
        no_diag = [w_in["W"+str(i)] - np.diag(np.diag(w_in["W"+str(i)])) for i in range(len(w_in))]
        row_sums = [np.repeat(np.sum(w, axis=1).reshape(-1, 1), w.shape[0], axis=1) for w in no_diag]
        return np.array(no_diag)/np.array(row_sums)

    def info(self) -> str:
        raise NotImplementedError

    def generate_random_walks(self, w_in, w_f):
        Z = self.__generate_normalization(w_in)
        N = len(self.get_graph().nodes)
        random_walks = np.empty((self.__gamma*N, self.__T))
        for i in range(self.__gamma):
            for j in range(N):
                random_walks[(i*N)+j, 0] = j
                rw_length = 1
                cur_layer = 0
                cur_v = j
                while rw_length <= self.__T:
                    if np.random.random() > self.__q:
                        if cur_layer == 0:
                            cur_layer = 1
                            cur_w = Z[cur_layer]
                            if np.any(np.isnan(cur_w[cur_v, :])):
                                cur_layer = 0
                        elif cur_layer == self.__k:
                            cur_layer = self.__k-1
                            cur_w = Z[cur_layer]
                            if np.any(np.isnan(cur_w[cur_v, :])):
                                cur_layer = self.__k
                        else:
                            p = w_f[cur_layer]/(w_f[cur_layer]+1)
                            if np.random.random() > p:
                                cur_layer += 1
                                cur_w = Z[cur_layer]
                                if np.any(np.isnan(cur_w[cur_v, :])):
                                    cur_layer -= 1
                            else:
                                cur_layer -= 1
                                cur_w = Z[cur_layer]
                                if np.any(np.isnan(cur_w[cur_v, :])):
                                    cur_layer += 1
                    cur_w = Z[cur_layer]
                    next_v = np.random.choice(np.arange(N), p=cur_w[cur_v, :])
                    random_walks[(i*N)+j, rw_length-1] = next_v
                    cur_v = next_v
                    rw_length += 1
        return random_walks.astype(int).astype(str).tolist()

    def embed(self, iter_num=1000, alpha=0.1, min_alpha=0.01) -> ndarray:
        matrix_dict = self.generate_similarity_matrices()
        w_in, w_f = self.generate_multigraph_edges(matrix_dict)
        rw = self.generate_random_walks(w_in, w_f)
        self.__model = Word2Vec(alpha=alpha, min_alpha=min_alpha,
                                min_count=0, size=self.__d, window=self.__window, sg=1, hs=1, negative=0)
        self.__model.build_vocab(sentences=rw)
        self.__model.train(sentences=rw, total_examples=len(rw), total_words=len(self.get_graph().nodes),
                           epochs=iter_num)
        return self.__model.wv.vectors
