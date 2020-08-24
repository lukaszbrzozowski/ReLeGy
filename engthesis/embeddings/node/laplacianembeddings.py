from typing import Dict, Any, Union, Callable

from networkx import laplacian_matrix, to_numpy_matrix
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from engthesis.model.base import Model
import numpy as np


class LaplacianEmbeddings(Model):

    def __init__(self, graph, d=2) -> None:
        """

        :rtype: object
        """
        super().__init__(graph)
        self.__A: csr_matrix = to_numpy_matrix(self.get_graph())
        self.__d = d

    def info(self) -> str:
        return "To be implemented"

    def embed(self) -> np.ndarray:
        """
            Find laplacian eigenmap encoding of graph nodes simulating similarity measure in R^m space

            Keyword arguments:
            graph -- graph with n nodes describing node connections
            A -- similarity function matrix - preferably sparse matrix
            m -- output dimension
        """
        n = len(self.get_graph().nodes)
        L = laplacian_matrix(self.get_graph())
        D = L + self.__A
        Y0 = np.random.rand(n, self.__d).reshape(-1)
        Id = np.eye(self.__d)
        flat = lambda f: lambda Y_flat: f(Y_flat.reshape(n, self.__d)).reshape(-1)
        func = lambda Y: np.trace(Y.T @ L @ Y)
        der = lambda Y: (2 * L @ Y).reshape(-1)

        eq_cons: Dict[str, Union[str, Callable[[Any], Any]]] = {"type": "eq",
                                                                "fun": flat(lambda Y: np.sum((Y.T @ D @ Y - Id) ** 2)),
                                                                "jac": flat(lambda Y: 2 * D @ Y)}

        res = minimize(flat(func),
                       Y0,
                       method='SLSQP',
                       jac=flat(der),
                       constraints=eq_cons,
                       options={'ftol': 1e-9, 'disp': True, 'maxiter': 200})
        return res.x.reshape(n, self.__d)
