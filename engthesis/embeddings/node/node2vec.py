from typing import Any

from networkx import to_numpy_matrix
import numpy as np
from numpy import matrix, ndarray
from gensim.models import Word2Vec

from engthesis.model.base import Model


class Node2Vec(Model):
    __A: matrix
    __d: int
    __T: int
    __gamma: int
    __window: int
    __p: float
    __q: float

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
        p - parameter of random walks, default 1
        q - parameter of random walks, default 1
        :return:
        """
        self.__gamma = parameters["gamma"] if "gamma" in parameters else 1
        self.__T = parameters["T"] if "T" in parameters else 2
        self.__window = parameters["window"] if "window" in parameters else 5
        self.__d = parameters["d"] if "d" in parameters else 2

        self.__p = parameters["p"] if "p" in parameters else 1
        self.__q = parameters["q"] if "q" in parameters else 1

    def generate_random_walks(self) -> Any:
        G = self.get_graph()
        N: int = len(G.nodes)
        A: matrix = to_numpy_matrix(G)
        p: float = self.__p
        q: float = self.__q
        random_walks: ndarray = np.empty((0, self.__T))
        for i in range(self.__gamma):
            # random_walk_matrix contains random walks for 1 iteration of i
            random_walk_matrix = np.empty((N, self.__T))
            vertices = np.arange(N)
            random_walk_matrix[:, 0] = vertices
            probabilities = np.multiply(A, 1 / np.repeat(np.sum(A, axis=1), [N]).reshape(N, N))
            # generation 2nd step of the random walks with uniform probability
            next_vertices = [np.random.choice(vertices, size=1, p=np.asarray(probabilities[it, :]).reshape(-1))[0]
                             for it in range(N)]
            random_walk_matrix[:, 1] = next_vertices
            for j in range(2, self.__T):
                probabilities_mask = np.ones((N, N))
                # generating mask on previous vertices in the random walks
                mask_p = np.identity(N)[vertices, :]
                # generating mask on the vertices which are approachable from current vertices, unapproachable from
                # previous vertices and are not the previous vertices
                mask_q = np.logical_and(A[next_vertices, :] > 0,
                                        np.logical_and(np.logical_and(A @ A > 0, A <= 0),
                                                       np.logical_not(np.identity(N)))[vertices, :])
                # modifying probabilities' mask accordingly
                probabilities_mask = np.where(mask_p, probabilities_mask / p, probabilities_mask)
                probabilities_mask = np.where(mask_q, probabilities_mask / q, probabilities_mask)
                probabilities = np.multiply(A[next_vertices, :], probabilities_mask)
                # normalizing probabilities
                probabilities = np.multiply(probabilities, 1 / np.repeat(np.sum(probabilities, axis=1),
                                                                         [N]).reshape(N, N))
                cur_next_vertices = [np.random.choice(np.arange(N), size=1,
                                                      p=np.asarray(probabilities[it, :]).reshape(-1))[0]
                                     for it in range(N)]
                random_walk_matrix[:, j] = cur_next_vertices
                vertices, next_vertices = next_vertices, cur_next_vertices

            random_walks = np.concatenate((random_walks, random_walk_matrix))

        return random_walks.astype(int).astype(str).tolist()

    def info(self) -> str:
        return "TBI"

    def embed(self, iter_num=1000, alpha=0.1, min_alpha=0.01, negative=5) -> ndarray:
        """
        The main embedding function of the node2vec model
        :param iter_num: Number of epochs
        :param alpha: Learning rate
        :param min_alpha: Minimal value of the learning rate; if defined, alpha decreases linearly
        :param negative: number of negative samples. If 0, no negative sampling is used
        :return: ndarray
        """
        rw = self.generate_random_walks()
        self.__model = Word2Vec(alpha=alpha, min_alpha=min_alpha,
                                min_count=0, size=self.__d, window=self.__window, sg=1, hs=1, negative=negative)
        self.__model.build_vocab(sentences=rw)
        self.__model.train(sentences=rw, total_examples=len(rw), total_words=len(self.get_graph().nodes),
                           epochs=iter_num)
        wv = self.__model.wv
        Z = np.empty((len(rw) // self.__gamma, self.__d))
        for i in range(Z.shape[0]):
            Z[i, :] = wv[str(i)]
        return Z
