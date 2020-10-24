from typing import Any

import numpy as np
from gensim.models import Word2Vec
from networkx import to_numpy_matrix, Graph
from numpy import matrix, ndarray

from engthesis.model import Model


class Node2Vec(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 T: int = 40,
                 window_size: int = 5,
                 gamma: int = 1,
                 p: float = 1,
                 q: float = 1) -> None:
        """
        The initialization method of the Node2Vec model.
        :param graph: The graph to be embedded
        :param d: dimensionality of the embedding vectors
        :param T: Length of the random walks
        :param gamma: Number of times a random walk is started from each vertex
        :param window_size: Window size for the SkipGram model
        :param p: Parameter of the biased random walks
        :param q: Parameter of the biased random walks
        """
        super().__init__(graph)
        self.__gamma: int = gamma
        self.__T: int = T
        self.__window: int = window_size
        self.__d: int = d
        self.__p: float = p
        self.__q: float = q

        self.__model = None

    def generate_random_walks(self) -> Any:
        G = self.get_graph()
        N: int = len(G.nodes)
        A: matrix = to_numpy_matrix(G, np.arange(len(G.nodes)))
        p: float = self.__p
        q: float = self.__q
        random_walks: ndarray = np.empty((N * self.__gamma, self.__T))
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
                normalized_probabilities = np.multiply(probabilities, 1 / np.repeat(np.sum(probabilities, axis=1),
                                                                                    [N]).reshape(N, N))
                cur_next_vertices = [np.random.choice(np.arange(N), size=1,
                                                      p=np.asarray(normalized_probabilities[it, :]).reshape(-1))[0]
                                     for it in range(N)]
                random_walk_matrix[:, j] = cur_next_vertices
                vertices, next_vertices = next_vertices, cur_next_vertices

            random_walks[(i * N):((i + 1) * N), :] = random_walk_matrix

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
        :return: Embedding of the graph into R^d
        """
        rw = self.generate_random_walks()
        self.__model = Word2Vec(alpha=alpha, min_alpha=min_alpha,
                                min_count=0, size=self.__d, window=self.__window, sg=1, hs=0, negative=negative)
        self.__model.build_vocab(sentences=rw)
        self.__model.train(sentences=rw, total_examples=len(rw), total_words=len(self.get_graph().nodes),
                           epochs=iter_num)
        wv = self.__model.wv
        Z = np.empty((len(rw) // self.__gamma, self.__d))
        for i in range(Z.shape[0]):
            Z[i, :] = wv[str(i)]
        return Z
