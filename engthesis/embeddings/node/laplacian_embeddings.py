from networkx import laplacian_matrix, adjacency_matrix
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from engthesis.model.base import Model
import numpy as np

class LaplacianEmbeddings(Model):

    __A: csr_matrix
    __m: int

    def __init__(self, graph, **kwargs) -> None:
        super().__init__(graph)
        self.initializeParameters(kwargs)

    def initializeParameters(self, parameters) -> None:
        """
        :param parameters: dictionary of model parameters
        A - weight matrix, default is adjacency matrix
        m - dimension of returned R^m vectors
        :return: None
        """
        self.__A = parameters["A"] if "A" in parameters else adjacency_matrix(self.getGraph())
        self.__m = parameters["m"] if "m" in parameters else 2

    def info(self) -> str:
        return "mama"

    def embed(self) -> np.ndarray:
        """
            Find laplacian eigenmap encoding of graph nodes simulating similarity measure in R^m space

            Keyword arguments:
            graph -- graph with n nodes describing node connections
            A -- similarity function matrix - preferably sparse matrix
            m -- output dimension
        """
        n = len(self.getGraph().nodes)
        D = laplacian_matrix(self.getGraph()) + adjacency_matrix(self.getGraph())
        L = D - self.__A
        Y0 = np.random.rand(n, self.__m).reshape(-1)
        I = np.eye(self.__m)
        flat = lambda f: lambda Y_flat: f(Y_flat.reshape(n, self.__m)).reshape(-1)
        func = lambda Y: np.trace(Y.T @ L @ Y)
        der = lambda Y: (2 * L @ Y).reshape(-1)

        eq_cons = {"type": "eq",
                   "fun": flat(lambda Y: np.sum((Y.T @ D @ Y - I) ** 2)),
                   "jac": flat(lambda Y: 2 * D @ Y)}

        res = minimize(flat(func),
                       Y0,
                       method='SLSQP',
                       jac=flat(der),
                       constraints=[eq_cons],
                       options={'ftol': 1e-9, 'disp': True, 'maxiter': 200})
        return res.x