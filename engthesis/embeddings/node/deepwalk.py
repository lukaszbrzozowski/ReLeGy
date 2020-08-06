from typing import Any

from networkx import to_numpy_matrix
import numpy as np
from numpy import matrix, ndarray
from gensim.models import Word2Vec

from engthesis.model.base import Model


class DeepWalk(Model):
    __A: matrix
    __d: int
    __T: int
    __gamma: int
    __window: int

    def __init__(self, graph, **kwargs) -> None:
        """

        :rtype: object
        """
        super().__init__(graph)
        self.initialize_parameters(kwargs)
        self.__model = None

    def initialize_parameters(self, parameters) -> None:
        """

        :param parameters:  dictionary of model parameters
        d - dimension of returned vectors
        T - length of random walk
        gamma - number of random walks starting in a single vertex, default 5
        window - window size of the SkipGram model
        :return:
        """
        self.__d = parameters["d"] if "d" in parameters else 2
        self.__T = parameters["T"] if "T" in parameters else 2
        self.__gamma = parameters["gamma"] if "gamma" in parameters else 1
        self.__window = parameters["window"] if "window" in parameters else 5

    def generate_random_walks(self) -> Any:
        G = self.get_graph()
        N = len(G.nodes)
        A = to_numpy_matrix(G)
        # P - 1-step transition probability matrix
        P = np.diag(1 / np.sum(A, axis=1).A1) @ A
        random_walks = np.empty((0, self.__T))
        for i in range(self.__gamma):
            random_walk_matrix = np.empty((N, self.__T))
            random_walk_matrix[:, 0] = np.arange(N)
            for j in range(1, self.__T):
                next_vertices = [np.random.choice(A.shape[1], size=1, p=np.asarray(P[it, :]).reshape(-1))[0] for it
                                 in range(A.shape[0])]
                random_walk_matrix[:, j] = next_vertices
                P = P[next_vertices, :]

            random_walks = np.concatenate((random_walks, random_walk_matrix))
        return random_walks.astype(int).astype(str).tolist()

    def info(self) -> str:
        return "TBI"

    def embed(self, iter_num=1000, alpha=0.1, min_alpha=0.01) -> ndarray:
        """
        The main embedding function of the DeepWalk model.
        :param iter_num: Number of epochs
        :param alpha: Learning rate
        :param min_alpha: Minimal value of the learning rate; if defined, alpha decreases linearly
        :return: ndarray
        """
        rw = self.generate_random_walks()
        self.__model = Word2Vec(alpha=alpha, min_alpha=min_alpha,
                                min_count=0, size=self.__d, window=self.__window, sg=1, hs=1, negative=0)
        self.__model.build_vocab(sentences=rw)
        self.__model.train(sentences=rw, total_examples=len(rw), total_words=len(self.get_graph().nodes),
                           epochs=iter_num)
        wv = self.__model.wv
        Z = np.empty((len(rw)//self.__gamma, self.__d))
        for i in range(Z.shape[0]):
            Z[i, :] = wv[str(i)]
        return Z


