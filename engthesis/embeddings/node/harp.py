from engthesis.model.base import Model
import numpy as np
from numpy import ndarray
import networkx as nx


class HARP(Model):
    def __init__(self, graph, **kwargs):
        __L: int
        __threshold: int
        __d: int
        __method: str

        super().__init__(graph)
        parameters = kwargs
        self.__threshold = parameters["threshold"] if "threshold" in parameters else 100
        self.__L = parameters["L"] if "L" in parameters else None
        if self.__L is not None:
            self.__threshold = None
        self.__d = parameters["d"] if "d" in parameters else 2
        self.__method = parameters["method"] if "method" in parameters else "DeepWalk"
        assert (self.__method in ["DeepWalk", "Node2Vec", "LINE"])

    @staticmethod
    def generate_convert_list(G):
        A = nx.to_numpy_array(G)
        N = A.shape[0]
        convert_list = np.array([np.arange(N), np.arange(N)]).T
        unq, inv, counts = np.unique(A, axis=0, return_inverse=True, return_counts=True)
        sim_ixes_list = np.array([np.arange(N)[inv == i] for i in np.arange(N)])
        ixs = np.sort(np.unique(inv))
        cur_ixs = ixs[counts > 1]
        cur_sim = sim_ixes_list[cur_ixs]

        for sim_ixes in cur_sim:
            length = len(sim_ixes)
            div_number = length // 2 if not length % 2 else length // 2 + 1
            pairs = np.array(np.array_split(sim_ixes, div_number))
            if len(pairs[-1]) == 1: pairs = pairs[:-1]
            stacked_pairs = np.vstack(pairs)
            convert_list[stacked_pairs[:, 0], 1] = stacked_pairs[:, 1]
        return convert_list

    @staticmethod
    def generate_star_collapsed_graph(G, convert_list):
        nG = G.copy()
        to_collapse = convert_list[convert_list[:, 0] != convert_list[:, 1]]
        for row in to_collapse:
            nG = nx.contracted_nodes(nG, row[1], row[0], self_loops=False)
        mapping = dict(zip(nG, range(len(G.nodes))))
        nG = nx.relabel_nodes(nG, mapping)
        return nG, mapping

    @staticmethod
    def get_mapping_from_dict(cl_vec, mapping):
        nv = np.empty(cl_vec.shape[0])
        for i in range(nv.shape[0]):
            nv[i] = mapping[cl_vec[i]]
        return nv

    @staticmethod
    def generate_edge_collapsed_graph(G):
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

    def edge_collapse(self, G):

        nG, convert_list, mapping = self.generate_edge_collapsed_graph(G)
        nv = self.get_mapping_from_dict(convert_list[:, 1], mapping)
        return nG, np.array([np.arange(len(G.nodes)), nv]).T

    def star_collapse(self, G):

        convert_list = self.generate_convert_list(G)

        nG, mapping = self.generate_star_collapsed_graph(G, convert_list)

        nv = self.get_mapping_from_dict(convert_list[:, 1], mapping)

        return nG, np.array([np.arange(len(G.nodes)), nv]).T

    def edge_star_collapse(self, G):
        nG0, cl0 = self.star_collapse(G)
        nG, cl1 = self.edge_collapse(nG0)
        cl = np.array([np.arange(len(G.nodes)), cl1[:, 1][cl0[:, 1].astype(int)]]).T
        return nG, cl

    def generate_contracted_graphs(self):
        G = self.get_graph()
        transition_matrix = np.arange(len(G.nodes)).reshape(-1, 1)
        graph_dict = {"G0": G}
        if self.__L is not None:
            for i in range(self.__L):
                G, cl = self.edge_star_collapse(G)
                graph_dict["G"+str(i+1)] = G
                cln = cl[:, 1][transition_matrix[:, -1].astype(int)].reshape(-1, 1)
                transition_matrix = np.concatenate([transition_matrix, cln], axis=1)
                if np.all(cln == cln[0]):
                    print("The contraction stopped early - the graph was contracted to a single node")
                    break
            return graph_dict, transition_matrix
        i = 1
        while len(G.nodes) > self.__threshold:
            G, cl = self.edge_star_collapse(G)
            graph_dict["G" + str(i)] = G
            i += 1
            cln = cl[:, 1][transition_matrix[:, -1].astype(int)].reshape(-1, 1)
            transition_matrix = np.concatenate([transition_matrix, cln], axis=1)
        return graph_dict, transition_matrix

    def info(self) -> str:
        return "TBI"

    def embed(self) -> ndarray:
        return np.array([])
