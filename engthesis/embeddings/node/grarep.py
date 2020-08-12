import numpy as np
from numpy import matrix, ndarray
from engthesis.model.base import Model
from networkx import to_numpy_matrix
import copy
class GraRep(Model):
    __A: matrix
    __d: int
    __K: int
    __lmbd: float
    __isEmbed: bool

    def __init__(self, graph, **kwargs) -> None:
        """

        :rtype: object
        """
        super().__init__(graph)
        self.__dict = {}
        self.initialize_parameters(kwargs)

    def initialize_parameters(self, parameters) -> None:
        """

        :param parameters: dictionary of model parameters
        A - weight matrix, default is adjacency matrix
        d - dimension of returned vectors for each k in [1, K]. The resulting matrix has K*d columns
        K - maximal order of information captured
        lmbd - regularization parameter
        :return: None
        """
        self.__A = parameters["A"] if "A" in parameters else to_numpy_matrix(self.get_graph())
        self.__d = parameters["d"] if "d" in parameters else 2
        self.__K = parameters["K"] if "K" in parameters else 1
        self.__lmbd = parameters["lmbd"] if "lmbd" in parameters else 1
        self.__isEmbed = False

    def info(self) -> str:
        return "To be implemented"

    def embed(self) -> ndarray:
        G = self.get_graph()
        N = len(G.nodes)
        beta = self.__lmbd/N
        returnW = None
        for i in range(1, self.__K+1):
            if i == 1:
                D = np.diag(np.sum(self.__A, axis=0).A1)
                D1 = np.linalg.inv(D)
                S = D1 @ self.__A
                S_current = copy.deepcopy(S)
            else:
                S_current = S_current @ S
            gamma = np.repeat(np.sum(S_current, axis=0), [self.__A.shape[0]], axis=0)
            Y = np.log(np.multiply(S_current, 1/gamma))-np.log(beta)
            X = Y
            X[X < 0] = 0
            U, D, VT = np.linalg.svd(X)
            Ud = U[:, :self.__d]
            Dd = np.diag(D)[:self.__d, :self.__d]
            W = Ud @ np.sqrt(Dd)
            returnW = W if returnW is None else np.concatenate((returnW, W), axis=1)
            self.__dict["W"+str(i)] = W
        self.__isEmbed = True
        return returnW

    def get_matrix_dict(self) -> dict:
        if not self.__isEmbed:
            print("The graph has not been embedded yet")
        return self.__dict




