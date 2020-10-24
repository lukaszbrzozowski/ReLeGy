from engthesis.model import Model

from typing import Dict, Any, Union, Callable

import numpy as np
from networkx import laplacian_matrix, to_numpy_array, Graph
from scipy.optimize import minimize


class LaplacianEigenmaps(Model):

    def __init__(self,
                 graph: Graph):
        super().__init__(graph)

        self.__N = None
        self.__Y0 = None
        self.__flat = None
        self.__func = None
        self.__der = None
        self.__d = None
        self.__eq_cons = None
        self.__result = None

        self.__initialized = False
        self.__fitted = False

    def initialize(self,
                   d: int = 2):
        self.__d = d
        self.__N = len(self.get_graph().nodes)
        self.__Y0 = np.random.rand(self.__N, self.__d).reshape(-1)

        L = laplacian_matrix(self.get_graph()).toarray()
        A = to_numpy_array(self.get_graph(), nodelist=np.arange(self.__N))
        D = L + A
        Id = np.eye(self.__d)

        self.__flat = lambda f: lambda Y_flat: f(Y_flat.reshape(self.__N, self.__d)).reshape(-1)
        self.__func = lambda Y: np.trace(Y.T @ L @ Y)
        self.__der = lambda Y: (2 * L @ Y).reshape(-1)

        self.__eq_cons: Dict[str, Union[str, Callable[[Any], Any]]] = {"type": "eq",
                                                                       "fun": self.__flat(
                                                                           lambda Y: np.sum((Y.T @ D @ Y - Id) ** 2)),
                                                                       "jac": self.__flat(lambda Y: 2 * D @ Y)}

        self.__initialized = True
        self.__fitted = False

    def info(self) -> str:
        raise NotImplementedError

    def fit(self,
            ftol: float = 1e-7,
            verbose: bool = True,
            num_iter: int = 200):
        """
        The embedding method of the Laplacian Eigenmaps.
        :param ftol: Precision parameter of the optimisation process. Default 1e-7
        :param verbose: Whether to print optimisation results after the embedding
        :param maxiter: Maximal number of iterations of the optimisation process
        :return: The graph embedding in R^d
        """

        if not self.__initialized:
            raise Exception("The method 'initialize' must be called before fitting")

        res = minimize(self.__flat(self.__func),
                       self.__Y0,
                       method='SLSQP',
                       jac=self.__flat(self.__der),
                       constraints=self.__eq_cons,
                       options={'ftol': ftol, 'disp': verbose, 'maxiter': num_iter})

        self.__result = res.x.reshape(-1, self.__d)
        self.__fitted = True

    def embed(self) -> np.ndarray:
        if not self.__initialized:
            raise Exception("The methods 'initialize' and 'fit' must be called before embedding")
        if not self.__fitted:
            raise Exception("The method 'fit' must be called before embedding")
        return self.__result

    @staticmethod
    def fast_embed(graph: Graph,
                   d: int = 2,
                   num_iter: int = 200,
                   ftol: float = 1e-7,
                   fit_verbose: bool = True) -> np.ndarray:

        LE = LaplacianEigenmaps(graph)

        LE.initialize(d=d)

        LE.fit(num_iter=num_iter,
               ftol=ftol,
               verbose=fit_verbose)

        return LE.embed()