import numpy as np
from numpy import matrix, ndarray
from engthesis.model.base import Model
from networkx import to_numpy_matrix


class GraRep(Model):

    def __init__(self, graph, **kwargs) -> None:
        """

        :rtype: object
        """
        __A: matrix
        __d: int
        __K: int
        __lmbd: float
        __isEmbed: bool

        super().__init__(graph)
        parameters = kwargs
        self.__A = parameters["A"] if "A" in parameters else to_numpy_matrix(self.get_graph())
        self.__d = parameters["d"] if "d" in parameters else 2
        self.__K = parameters["K"] if "K" in parameters else 1
        self.__lmbd = parameters["lmbd"] if "lmbd" in parameters else 1
        self.__isEmbed = False
        self.__modelDict = {}

    def info(self) -> str:
        return "To be implemented"

    def embed(self) -> ndarray:
        G = self.get_graph()
        N = len(G.nodes)
        beta = self.__lmbd/N
        D1 = np.diag(1/np.sum(self.__A, axis=0).A1)
        S = D1 @ self.__A
        S_current = np.identity(N)
        for i in range(1, self.__K+1):
            S_current = S_current @ S
            gamma = np.repeat(np.sum(S_current, axis=0), [N], axis=0)
            Y = np.log(S/gamma)-np.log(beta)
            X = Y
            X[X < 0] = 0
            U, D, VT = np.linalg.svd(X)
            Ud = U[:, :self.__d]
            Dd = np.diag(D)[:self.__d, :self.__d]
            W = Ud @ np.sqrt(Dd)
            self.__modelDict["W"+str(i)] = W
        self.__isEmbed = True
        returnW = np.concatenate([self.__modelDict[x] for x in sorted(self.__modelDict)], 1)
        return returnW

    def get_matrix_dict(self) -> dict:
        if not self.__isEmbed:
            print("The graph has not been embedded yet")
        return self.__modelDict




