from engthesis.model.base import Model
from engthesis.embeddings.node.deepwalk import DeepWalk
from engthesis.embeddings.node.node2vec import Node2Vec
import numpy as np
from numpy import ndarray
import networkx as nx
from networkx import Graph
from gensim.models import Word2Vec
from six import iteritems


class HARP(Model):
    def __init__(self,
                 graph: Graph,
                 d: int = 2,
                 method: str = "DeepWalk",
                 threshold: int = 100,
                 L: int = None,
                 T: int = 40,
                 window_size: int = 5,
                 gamma: int = 1,
                 p: float = 1,
                 q: float = 1,
                 verbose: bool = 1):
        """
        The initialization method of the HARP model.
        :param graph: The graph to be embedded
        :param d: dimensionality of the embedding vectors
        :param method: The method to use in representation learning after the graph coarsening.
        Possible are "DeepWalk", "Node2Vec" and #TODO add LINE to HARP
        "LINE"
        :param threshold: The maximal number of vertices of the coarsened graph. Not used if L is passed.
        :param L: Number of iterations of the graph coarsening.
        :param T: Length of the random walks (DeepWalk and Node2Vec)
        :param gamma: Number of times a random walk is started from each vertex (DeepWalk and Node2Vec)
        :param window_size: Window size for the SkipGram model (DeepWalk and Node2Vec)
        :param p: Parameter of the biased random walks (Node2Vec)
        :param q: Parameter of the biased random walks (Node2Vec)
        :param verbose: Verbosity of the graph coarsening
        """

        super().__init__(graph)
        self.__threshold: int = threshold
        self.__L: int = L
        if self.__L is not None:
            self.__threshold = None
        self.__d: int = d
        self.__method: str = method
        assert (self.__method in ["DeepWalk", "Node2Vec", "LINE"])

        self.__T: int = T
        self.__gamma: int = gamma
        self.__window: int = window_size
        self.__verbose: bool = verbose

        self.__p: float = p
        self.__q: float = q
        self.__model = None

    @staticmethod
    def __generate_convert_list(G):
        A = nx.to_numpy_array(G)
        N = A.shape[0]
        convert_list = np.array([np.arange(N), np.arange(N)]).T
        unq, inv, counts = np.unique(A, axis=0, return_inverse=True, return_counts=True)
        sim_ixs_list = np.array([np.arange(N)[inv == i] for i in np.arange(N)])
        ixs = np.sort(np.unique(inv))
        cur_ixs = ixs[counts > 1]
        cur_sim = sim_ixs_list[cur_ixs]

        for sim_ixs in cur_sim:
            length = len(sim_ixs)
            div_number = length // 2 if not length % 2 else length // 2 + 1
            pairs = np.array(np.array_split(sim_ixs, div_number))
            if len(pairs[-1]) == 1:
                pairs = pairs[:-1]
            stacked_pairs = np.vstack(pairs)
            convert_list[stacked_pairs[:, 0], 1] = stacked_pairs[:, 1]
        return convert_list

    @staticmethod
    def __generate_star_collapsed_graph(G, convert_list):
        nG = G.copy()
        to_collapse = convert_list[convert_list[:, 0] != convert_list[:, 1]]
        for row in to_collapse:
            nG = nx.contracted_nodes(nG, row[1], row[0], self_loops=False)
        mapping = dict(zip(nG, range(len(G.nodes))))
        nG = nx.relabel_nodes(nG, mapping)
        return nG, mapping

    @staticmethod
    def __get_mapping_from_dict(cl_vec, mapping):
        nv = np.empty(cl_vec.shape[0])
        for i in range(nv.shape[0]):
            nv[i] = mapping[cl_vec[i]]
        return nv

    @staticmethod
    def __generate_edge_collapsed_graph(G):
        mx_matching = nx.algorithms.matching.maximal_matching(G)
        N = len(G.nodes)
        conv_list = np.array([np.arange(N), np.arange(N)]).T
        nG = G.copy()
        for e in mx_matching:
            nG = nx.contracted_nodes(nG, e[1], e[0], self_loops=False)
            conv_list[e[0], 1] = e[1]
        mapping = dict(zip(nG, range(len(G.nodes))))
        nG = nx.relabel_nodes(nG, mapping)

        return nG, conv_list, mapping

    def __edge_collapse(self, G):

        nG, convert_list, mapping = self.__generate_edge_collapsed_graph(G)
        nv = self.__get_mapping_from_dict(convert_list[:, 1], mapping)
        return nG, np.array([np.arange(len(G.nodes)), nv]).T

    def __star_collapse(self, G):

        convert_list = self.__generate_convert_list(G)

        nG, mapping = self.__generate_star_collapsed_graph(G, convert_list)

        nv = self.__get_mapping_from_dict(convert_list[:, 1], mapping)

        return nG, np.array([np.arange(len(G.nodes)), nv]).T

    def edge_star_collapse(self, G):
        nG_star, cl_star = self.__star_collapse(G)
        nG_edge, cl_edge = self.__edge_collapse(nG_star)
        cl = np.array([np.arange(len(G.nodes)), cl_edge[:, 1][cl_star[:, 1].astype(int)]]).T
        return nG_edge, cl

    def generate_collapsed_graphs(self):
        G = self.get_graph()
        transition_matrix = np.arange(len(G.nodes)).reshape(-1, 1)
        graph_dict = {"G0": G}
        if self.__L is not None:
            for i in range(self.__L):
                G, cl = self.edge_star_collapse(G)
                graph_dict["G"+str(i+1)] = G
                cln = cl[:, 1][transition_matrix[:, -1].astype(int)].reshape(-1, 1)
                transition_matrix = np.concatenate([transition_matrix, cln], axis=1)
                if self.__verbose:
                    print("Collapsed graph nr " + str(i+1))
                    print("Number of nodes: " + str(len(G.nodes)))
                    print("Number of edges: " + str(len(G.edges)))
                if np.all(cln == cln[0]):
                    print("The collapsion stopped early - the graph was collapsed to a single node")
                    break
            return graph_dict, transition_matrix
        i = 1
        while len(G.nodes) > self.__threshold:
            G, cl = self.edge_star_collapse(G)
            graph_dict["G" + str(i)] = G
            i += 1
            cln = cl[:, 1][transition_matrix[:, -1].astype(int)].reshape(-1, 1)
            transition_matrix = np.concatenate([transition_matrix, cln], axis=1)
            if self.__verbose:
                print("Collapsed graph nr " + str(i - 1))
                print("Number of nodes: " + str(len(G.nodes)))
                print("Number of edges: " + str(len(G.edges)))
        return graph_dict, transition_matrix

    def info(self) -> str:
        return "TBI"

    def __get_weights_from_model(self):
        wv = self.__model.wv
        weight_matrix = np.empty((len(wv.vocab.keys()), self.__d+1))
        i = 0
        for word, vocab in sorted(iteritems(wv.vocab)):
            row = wv.syn0[vocab.index]
            weight_matrix[i, 0] = word
            weight_matrix[i, 1:] = row
            i += 1
        return weight_matrix

    def __generate_new_weights(self, weight_matrix, transition_matrix):
        sorter = np.argsort(weight_matrix[:, 0])
        permutation = sorter[np.searchsorted(weight_matrix[:, 0], transition_matrix[:, 1], sorter=sorter)]
        assert(np.all(weight_matrix[:, 0][permutation] == transition_matrix[:, 1]))
        new_weight_matrix = np.empty((transition_matrix.shape[0], self.__d+1))
        new_weight_matrix[:, 0] = transition_matrix[:, 0].astype(int)
        new_weight_matrix[:, 1:] = weight_matrix[permutation, 1:]
        sorted_new_weight_matrix = np.unique(new_weight_matrix, axis=0)
        return sorted_new_weight_matrix

    def __update_model_weights(self, weight_matrix):
        wv = self.__model.wv
        for row in weight_matrix:
            word = str(int(row[0]))
            assert(word in wv.vocab)
            wv.syn0[wv.vocab[word].index] = row[1:]

    def embed(self, iter_num=1000, alpha=0.1, min_alpha=0.01) -> ndarray:
        graph_dict, transition_matrix = self.generate_collapsed_graphs()
        graph_names = sorted(graph_dict.keys())
        last_graph = graph_dict[graph_names[-1]]
        random_walks = None
        if self.__method == "DeepWalk":
            dw = DeepWalk(last_graph, d=self.__d, T=self.__T, gamma=self.__gamma, window_size=self.__window)
            random_walks = dw.generate_random_walks()
        elif self.__method == "Node2Vec":
            n2v = Node2Vec(last_graph, d=self.__d, T=self.__T, gamma=self.__gamma, window_size=self.__window,
                           p=self.__p, q=self.__q)
            random_walks = n2v.generate_random_walks()
        assert(random_walks is not None)
        self.__model = Word2Vec(alpha=alpha, min_alpha=min_alpha,
                                min_count=0, size=self.__d, window=self.__window, sg=1, hs=1, negative=0)
        self.__model.build_vocab(sentences=random_walks)
        self.__model.train(sentences=random_walks, total_examples=len(random_walks), total_words=len(last_graph.nodes),
                           epochs=iter_num)
        weight_matrix = self.__get_weights_from_model()
        tr_matrix = transition_matrix[:, -2:]
        new_wm = self.__generate_new_weights(weight_matrix, tr_matrix)

        for i in np.arange(len(graph_names)-2, -1, -1):
            cur_graph = graph_dict[graph_names[i]]

            if self.__method == "DeepWalk":
                dw = DeepWalk(cur_graph, d=self.__d, T=self.__T, gamma=self.__gamma, window_size=self.__window)
                random_walks = dw.generate_random_walks()
            elif self.__method == "Node2Vec":
                n2v = Node2Vec(cur_graph, d=self.__d, T=self.__T, gamma=self.__gamma, window_size=self.__window,
                               p=self.__p, q=self.__q)
                random_walks = n2v.generate_random_walks()

            self.__model = Word2Vec(alpha=alpha, min_alpha=min_alpha,
                                    min_count=0, size=self.__d, window=self.__window, sg=1, hs=1, negative=0)

            self.__model.build_vocab(sentences=random_walks)
            self.__update_model_weights(new_wm)

            self.__model.train(sentences=random_walks, total_examples=len(random_walks),
                               total_words=len(cur_graph.nodes),
                               epochs=iter_num)
            if i > 0:
                weight_matrix = self.__get_weights_from_model()
                tr_matrix = transition_matrix[:, (i-1):(i+1)]
                new_wm = self.__generate_new_weights(weight_matrix, tr_matrix)

        return self.__model.wv.vectors
