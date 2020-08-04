from networkx import to_numpy_matrix
import numpy as np
from numpy import matrix, ndarray
from engthesis.model.base import Model

class HOPE(Model):
    __A: matrix
    __d: int
    __proximity: str
    __param: float
    __matrixDict: dict
    __isEmbed: bool

    def __init__(self, graph, **kwargs) -> None:
        """

        :rtype: object
        """
        super().__init__(graph)
        self.initialize_parameters(kwargs)

    def initialize_parameters(self, parameters) -> None:
        """

        :param parameters: dictionary of model parameters
        d - dimension of returned vectors
        proximity - method of proximity calculation. Available are: "Katz", "RPR", "CN", "AA". Default is "CN"
        param - parameter beta of Katz proximity or alpha of RPR - default 0.1. Ignored otherwise
        :return: None
        """
        self.__A = to_numpy_matrix(self.get_graph())
        self.__proximity = parameters["proximity"] if "proximity" in parameters else "Katz"
        self.__d = parameters["d"] if "d" in parameters else 2
        self.__param = parameters["param"] if "param" in parameters else 0.1
        self.__matrixDict = {}
        self.__isEmbed = False

    def info(self):
        return "To be implemented"

    def embed(self) -> ndarray:
        N = len(self.get_graph().nodes)
        if self.__proximity == "Katz" or self.__proximity not in ["RPR", "AA", "CN"]:
            par = self.__param
            Mg = np.identity(N) - par*self.__A
            Ml = par*self.__A
        elif self.__proximity == "RPR":
            par = self.__param
            D = np.diag(np.sum(self.__A, axis=0).A1)
            D1 = np.linalg.inv(D)
            P = D1 @ self.__A
            Mg = np.identity(N) - par*P
            Ml = (1-par)*np.identity(N)
        elif self.__proximity == "CN":
            Mg = np.identity(N)
            Ml = self.__A @ self.__A
        elif self.__proximity == "AA":
            D = np.diag([1/(np.sum(self.__A[:, i])+np.sum(self.__A[i, :])) for i in range(N)])
            Mg = np.identity(N)
            Ml = self.__A @ D @ self.__A
        #JDGSVD shall be implemented here
        S = np.linalg.inv(Mg) @ Ml
        U, D, VT = np.linalg.svd(S)
        Ds = np.sqrt(np.diag(D)[:self.__d, :self.__d])
        Us = U[:, :self.__d] @ Ds
        Ut = VT.T[:, :self.__d] @ Ds
        self.__matrixDict = {"Us": Us, "Ut": Ut}
        self.__isEmbed = True
        return Us.T @ Ut

    def getMatrixDict(self) -> dict:
        if not self.__isEmbed:
            print("The graph has not been embedded yet")
        return self.__matrixDict



