import copy

import numpy as np
from gensim.models import Word2Vec
from networkx import to_numpy_matrix, Graph
from numpy import ndarray
from six import iteritems

from engthesis.model.base import Model


class DeepWalk(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 T: int = 40,
                 gamma: int = 1,
                 window_size: int = 5,
                 random_state: int = None) -> None:
        """
        The initialization method of the DeepWalk model.
        :param graph: The graph to be embedded
        :param d: dimensionality of the embedding vectors
        :param T: Length of the random walks.
        :param gamma: Number of times a random walk is started from each vertex
        :param window_size: Window size for the SkipGram model
        """

        super().__init__(graph)
        self.__model = None
        self.__d: int = d
        self.__T: int = T
        self.__gamma: int = gamma
        self.__window: int = window_size
        self.__random_state: int = random_state

    def generate_random_walks(self) -> list:
        G = self.get_graph()
        N = len(G.nodes)
        A = to_numpy_matrix(G, nodelist=np.arange(len(G.nodes)))
        # P - 1-step transition probability matrix
        baseP = np.diag(1 / np.sum(A, axis=1).A1) @ A
        random_walks = np.empty((N * self.__gamma, self.__T))
        P = copy.deepcopy(baseP)
        for i in range(self.__gamma):
            random_walk_matrix = np.empty((N, self.__T))
            random_walk_matrix[:, 0] = np.arange(N)
            for j in range(1, self.__T):
                next_vertices = [np.random.choice(A.shape[1], size=1, p=np.asarray(P[it, :]).reshape(-1))[0] for it
                                 in range(A.shape[0])]
                random_walk_matrix[:, j] = next_vertices
                P = baseP[next_vertices, :]

            random_walks[(i * N):((i + 1) * N), :] = random_walk_matrix
        retList = random_walks.astype(int).astype(str).tolist()
        return retList

    def info(self) -> str:
        return "TBI"

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

    def embed(self, iter_num=1000, alpha=0.1, min_alpha=0.01) -> ndarray:
        """
        The main embedding function of the DeepWalk model.
        :param iter_num: Number of epochs
        :param alpha: Learning rate
        :param min_alpha: Minimal value of the learning rate; if defined, alpha decreases linearly
        :return: Embedding of the graph into R^d
        """
        if self.__random_state is not None:
            np.random.seed(self.__random_state)

        rw = self.generate_random_walks()
        self.__model = Word2Vec(alpha=alpha, min_alpha=min_alpha,
                                min_count=0, size=self.__d, window=self.__window,
                                sg=1, hs=1, negative=0, seed=self.__random_state)
        self.__model.build_vocab(sentences=rw)
        self.__model.train(sentences=rw, total_examples=len(rw), total_words=len(self.get_graph().nodes),
                           epochs=iter_num)
        return self.__get_weights_from_model()
