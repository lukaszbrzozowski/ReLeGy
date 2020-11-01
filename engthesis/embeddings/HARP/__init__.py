import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from networkx import Graph
from numpy import ndarray
from six import iteritems

from engthesis.embeddings.DeepWalk import DeepWalk
from engthesis.embeddings.Node2Vec import Node2Vec
from engthesis.model import Model


class HARP(Model):
    def __init__(self,
                 graph: Graph):

        super().__init__(graph)
        self.__threshold = None
        self.__method = None
        self.__L = None
        self.__T = None
        self.__gamma = None
        self.__verbose = None
        self.__p = None
        self.__q = None
        self.__graph_stack = None
        self.__transition_matrix = None

        self.__model = None

        self.__d = None
        self.__alpha = None
        self.__min_alpha = None
        self.__hs = None
        self.__negative = None
        self.__window = None

    @Model._init_in_init_model_fit
    def initialize(self,
                   method: str = "DeepWalk",
                   threshold: int = 100,
                   L: int = None,
                   T: int = 40,
                   gamma: int = 1,
                   p: int = 1,
                   q: int = 1,
                   verbose: bool = True):
        graph = self.get_graph()
        self.__method = method
        if threshold is not None:
            self.__threshold = threshold if len(graph.nodes) > threshold else len(graph.nodes) // 2
        if self.__L is not None:
            self.__threshold = None
        self.__L = L
        self.__T = T
        self.__gamma = gamma
        self.__p = p
        self.__q = q
        self.__verbose = verbose

        self.__graph_stack, self.__transition_matrix = self.generate_collapsed_graphs()

    @Model._init_model_in_init_model_fit
    def initialize_model(self,
                         d: int = 2,
                         alpha: float = 0.025,
                         min_alpha: float = 0.0001,
                         hs: int = 1,
                         negative: int = 0,
                         window: int = 5):
        self.__d = d
        self.__alpha = alpha
        self.__min_alpha = min_alpha
        self.__hs = hs
        self.__negative = negative
        self.__window = window

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

    def __generate_with_L(self, G, graph_stack, transition_matrix):
        if self.__L is not None:
            for i in range(self.__L):
                G, cl = self.edge_star_collapse(G)
                graph_stack.append(G)
                cln = cl[:, 1][transition_matrix[:, -1].astype(int)].reshape(-1, 1)
                transition_matrix = np.concatenate([transition_matrix, cln], axis=1)
                if self.__verbose:
                    print("Collapsed graph nr " + str(i + 1))
                    print("Number of nodes: " + str(len(G.nodes)))
                    print("Number of edges: " + str(len(G.edges)))
                if np.all(cln == cln[0]):
                    print("The collapsion stopped early - the graph was collapsed to a single node")
                    break
            return graph_stack, transition_matrix

    def __generate_with_threshold(self, G, graph_stack, transition_matrix):
        i = 1
        while len(G.nodes) > self.__threshold:
            G, cl = self.edge_star_collapse(G)
            graph_stack.append(G)
            i += 1
            cln = cl[:, 1][transition_matrix[:, -1].astype(int)].reshape(-1, 1)
            transition_matrix = np.concatenate([transition_matrix, cln], axis=1)
            if self.__verbose:
                print("Collapsed graph nr " + str(i - 1))
                print("Number of nodes: " + str(len(G.nodes)))
                print("Number of edges: " + str(len(G.edges)))
        return graph_stack, transition_matrix

    def generate_collapsed_graphs(self):
        G = self.get_graph()
        transition_matrix = np.arange(len(G.nodes)).reshape(-1, 1)
        graph_stack = [G]

        # L is not None, threshold is None
        if self.__L is not None:
            return self.__generate_with_L(G, graph_stack, transition_matrix)
        elif self.__threshold is not None:
            return self.__generate_with_threshold(G, graph_stack, transition_matrix)
        else:
            raise ValueError("None of the number of graphs L or node number threshold were specified")

    def info(self) -> str:
        raise NotImplementedError

    def __generate_random_walks(self, G):
        random_walks = None
        if self.__method == "DeepWalk":
            dw = DeepWalk(G)
            dw.initialize(self.__T,
                          self.__gamma)
            random_walks = dw.get_random_walks()
        elif self.__method == "Node2Vec":
            n2v = Node2Vec(G)
            n2v.initialize(T=self.__T,
                           gamma=self.__gamma,
                           p=self.__p,
                           q=self.__q)
            random_walks = n2v.get_random_walks()
        return random_walks

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

    def __generate_new_weights(self, weight_matrix, transition_matrix):
        sorter = np.argsort(weight_matrix[:, 0])
        permutation = sorter[np.searchsorted(weight_matrix[:, 0], transition_matrix[:, 1], sorter=sorter)]
        assert (np.all(weight_matrix[:, 0][permutation] == transition_matrix[:, 1]))
        new_weight_matrix = np.empty((transition_matrix.shape[0], self.__d + 1))
        new_weight_matrix[:, 0] = transition_matrix[:, 0].astype(int)
        new_weight_matrix[:, 1:] = weight_matrix[permutation, 1:]
        sorted_new_weight_matrix = np.unique(new_weight_matrix, axis=0)
        return sorted_new_weight_matrix

    def __update_model_weights(self, weight_matrix):
        wv = self.__model.wv
        for row in weight_matrix:
            word = str(int(row[0]))
            assert (word in wv.vocab)
            wv.syn0[wv.vocab[word].index] = row[1:]

    @Model._fit_in_init_model_fit
    def fit(self,
            num_iter=1000):
        graph_stack, transition_matrix = self.__graph_stack, self.__transition_matrix

        last_graph = graph_stack.pop()  # We treat the last created graph separately,
        # because we don't update the weights

        random_walks = self.__generate_random_walks(last_graph)
        assert (random_walks is not None)

        self.__model = Word2Vec(alpha=self.__alpha,
                                min_alpha=self.__min_alpha,
                                min_count=1,
                                size=self.__d,
                                window=self.__window,
                                sg=1,
                                hs=self.__hs,
                                negative=self.__negative)

        self.__model.build_vocab(sentences=random_walks)

        self.__model.train(sentences=random_walks,
                           total_examples=len(random_walks),
                           epochs=num_iter)

        weight_matrix = self.__get_weights_from_model()
        tr_matrix = transition_matrix[:, -2:]
        new_wm = self.__generate_new_weights(weight_matrix, tr_matrix)

        for i, cur_graph in reversed(list(enumerate(graph_stack))):

            random_walks = self.__generate_random_walks(cur_graph)

            self.__model = Word2Vec(alpha=self.__alpha,
                                    min_alpha=self.__min_alpha,
                                    min_count=1,
                                    size=self.__d,
                                    window=self.__window,
                                    sg=1,
                                    hs=self.__hs,
                                    negative=self.__negative)

            self.__model.build_vocab(sentences=random_walks)

            self.__update_model_weights(new_wm)

            self.__model.train(sentences=random_walks,
                               total_examples=len(random_walks),
                               epochs=num_iter)

            if i > 0:  # We don't need to update matrices after the last graph iteration
                weight_matrix = self.__get_weights_from_model()
                tr_matrix = transition_matrix[:, (i - 1):(i + 1)]
                new_wm = self.__generate_new_weights(weight_matrix, tr_matrix)

    @Model._embed_in_init_model_fit
    def embed(self) -> ndarray:
        ret_matrix = np.empty((self.__N, self.__d), dtype="float32")
        for i in np.arange(self.__N):
            ret_matrix[i, :] = self.__model.wv[str(i)]
        return ret_matrix

    @staticmethod
    def fast_embed(graph: Graph,
                   threshold: int = 100,
                   L: int = None,
                   T: int = 40,
                   gamma: int = 1,
                   p: int = 1,
                   q: int = 1,
                   init_verbose: bool = True,
                   d: int = 2,
                   alpha: float = 0.025,
                   min_alpha: float = 0.0001,
                   hs: int = 1,
                   negative: int = 0,
                   window: int = 5,
                   num_iter=1000):
        harp = HARP(graph)
        harp.initialize(threshold=threshold,
                        L=L,
                        T=T,
                        gamma=gamma,
                        p=p,
                        q=q,
                        verbose=init_verbose)
        harp.initialize_model(d=d,
                              alpha=alpha,
                              min_alpha=min_alpha,
                              hs=hs,
                              negative=negative,
                              window=window)
        harp.fit(num_iter=num_iter)
        return harp.embed()