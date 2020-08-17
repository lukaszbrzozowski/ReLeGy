from engthesis.model.base import Model
from engthesis.embeddings.node.deepwalk import DeepWalk
from engthesis.embeddings.node.node2vec import Node2Vec
import numpy as np
from numpy import ndarray
import networkx as nx
from gensim.models import Word2Vec
from six import iteritems


class HARP(Model):
    def __init__(self, graph, **kwargs):

        __L: int
        __threshold: int
        __d: int
        __method: str
        __T: int
        __gamma: int
        __window: int
        __verbose: bool
        __p: float
        __q: float

        super().__init__(graph)
        parameters = kwargs
        self.__threshold = parameters["threshold"] if "threshold" in parameters else 100
        self.__L = parameters["L"] if "L" in parameters else None
        if self.__L is not None:
            self.__threshold = None
        self.__d = parameters["d"] if "d" in parameters else 2
        self.__method = parameters["method"] if "method" in parameters else "DeepWalk"
        assert (self.__method in ["DeepWalk", "Node2Vec", "LINE"])

        self.__T = parameters["T"] if "T" in parameters else 2
        self.__gamma = parameters["gamma"] if "gamma" in parameters else 1
        self.__window = parameters["window"] if "window" in parameters else 5
        self.__verbose = parameters["verbose"] if "verbose" in parameters else True

        self.__p = parameters["p"] if "p" in parameters else 1
        self.__q = parameters["q"] if "q" in parameters else 1
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
                    print("The contraction stopped early - the graph was collapsed to a single node")
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
        uqs = np.sort(np.unique(transition_matrix[:, 0]))
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
            dw = DeepWalk(last_graph, d=self.__d, T=self.__T, gamma=self.__gamma, window=self.__window)
            random_walks = dw.generate_random_walks()
        elif self.__method == "Node2Vec":
            n2v = Node2Vec(last_graph, d=self.__d, T=self.__T, gamma=self.__gamma, window=self.__window,
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
            dw = DeepWalk(cur_graph, d=self.__d, T=self.__T, gamma=self.__gamma, window=self.__window)

            if self.__method == "DeepWalk":
                dw = DeepWalk(cur_graph, d=self.__d, T=self.__T, gamma=self.__gamma, window=self.__window)
                random_walks = dw.generate_random_walks()
            elif self.__method == "Node2Vec":
                n2v = Node2Vec(cur_graph, d=self.__d, T=self.__T, gamma=self.__gamma, window=self.__window,
                               p=self.__p, q=self.__q)
                random_walks = n2v.generate_random_walks()

            self.__model = Word2Vec(alpha=alpha, min_alpha=min_alpha,
                                    min_count=0, size=self.__d, window=self.__window, sg=1, hs=1, negative=0)

            self.__model.build_vocab(sentences=random_walks)
            self.__update_model_weights(new_wm)

            self.__model.train(sentences=random_walks, total_examples=len(random_walks), total_words=len(cur_graph.nodes),
                               epochs=iter_num)
            if i > 0:
                weight_matrix = self.__get_weights_from_model()
                tr_matrix = transition_matrix[:, (i-1):(i+1)]
                new_wm = self.__generate_new_weights(weight_matrix, tr_matrix)

        return self.__model.wv.vectors
