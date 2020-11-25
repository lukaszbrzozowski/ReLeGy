from relegy.__base import Model

from typing import Dict, Any, Union, Callable

import numpy as np
import networkx as nx
from networkx import Graph
from numpy import ndarray
from scipy.optimize import minimize

init_verification = {"d": [(lambda x: x > 0, "'d' must be greater than 0.")]}

fit_verification = {"num_iter": [(lambda x: x > 0, "'num_iter' must be greater than 0.")],
                    "ftol": [(lambda x: x > 0, "'ftol' must be greater than 0.")]}

fast_embed_verification = Model.dict_union(init_verification, fit_verification)



class LaplacianEigenmaps(Model):
    """
    The Laplacian Eigenmaps method implementation. \n
    The details may be found in: \n
    M. Belkin and P. Niyogi. Laplacian eigenmaps and spectral techniques for embedding and clustering. In NIPS, 2002.
    """

    def __init__(self,
                 graph: Graph
                 ):
        """
        Laplacian Eigenmaps - constructor (step I).

        @param graph: The graph to be embedded. Nodes of the graph must be a sorted array from 0 to n-1, where n is
        the number of vertices.

        """
        super().__init__(graph)

        self.__N = None
        self.__Y0 = None
        self.__flat = None
        self.__func = None
        self.__der = None
        self.__d = None
        self.__eq_cons = None
        self.__result = None

    @Model._init_in_init_fit
    @Model._verify_parameters(rules_dict=init_verification)
    def initialize(self,
                   d: int = 2
                   ):
        """
        Laplacian Eigenmaps - initialization (step II) \n
        Generates the Laplacian matrix and prepares constraints for the optimization.

        @param d: The dimension of the embedding.
        """
        self.__d = d
        self.__N = len(self.get_graph().nodes)
        self.__Y0 = np.random.rand(self.__N, self.__d).reshape(-1)

        L = nx.laplacian_matrix(self.get_graph(), nodelist=np.arange(self.__N)).toarray()
        A = nx.to_numpy_array(self.get_graph(), nodelist=np.arange(self.__N))
        D = L + A
        Id = np.eye(self.__d)

        self.__flat = lambda f: lambda Y_flat: f(Y_flat.reshape(self.__N, self.__d)).reshape(-1)
        self.__func = lambda Y: np.trace(Y.T @ L @ Y)
        self.__der = lambda Y: (2 * L @ Y).reshape(-1)

        self.__eq_cons: Dict[str, Union[str, Callable[[Any], Any]]] = {"type": "eq",
                                                                       "fun": self.__flat(
                                                                           lambda Y: np.sum((Y.T @ D @ Y - Id) ** 2)),
                                                                       "jac": self.__flat(lambda Y: 2 * D @ Y)}

    @Model._fit_in_init_fit
    @Model._verify_parameters(rules_dict=fit_verification)
    def fit(self,
            num_iter: int = 200,
            ftol: float = 1e-7,
            verbose: bool = True
            ):
        """
        Laplacian Eigenmaps - fit (step III) \n
        Minimizes the loss function using scipy.minimize.

        @param ftol: Precision parameter of the optimisation process. Default 1e-7. Details may be found in
        scipy.minimize documentation.
        @param verbose: Whether to print optimisation results after the fitting.
        @param num_iter: Maximal number of iterations of the optimization process.
        """

        res = minimize(self.__flat(self.__func),
                       self.__Y0,
                       method='SLSQP',
                       jac=self.__flat(self.__der),
                       constraints=self.__eq_cons,
                       options={'ftol': ftol, 'disp': verbose, 'maxiter': num_iter})

        self.__result = res.x.reshape(-1, self.__d)

    @Model._embed_in_init_fit
    def embed(self) -> np.ndarray:
        """
        Laplacian Eigenmaps - embed (step IV) \n
        Returns the embedding.

        @return: The embedding matrix of the shape N x d.
        """

        return self.__result

    @staticmethod
    @Model._verify_parameters(rules_dict=fast_embed_verification)
    def fast_embed(graph: Graph,
                   d: int = 2,
                   num_iter: int = 200,
                   ftol: float = 1e-7,
                   fit_verbose: bool = True) -> ndarray:
        """
        Laplacian Eigenmaps - fast embed \n
        Performs the embedding in one step.

        @param graph: The graph to be embedded. Present in '__init__'
        @param d: The dimension of the embedding. Present in 'initialize'
        @param num_iter: Maximal number of iterations of the optimisation process. Present in 'fit'
        @param ftol: Precision parameter of the optimisation process. Present in 'fit'
        @param fit_verbose: Whether to print optimisation results after the fitting. Present in 'fit'
        @return: The embedding matrix of the shape N x d.
        """

        LE = LaplacianEigenmaps(graph)

        LE.initialize(d=d)

        LE.fit(num_iter=num_iter,
               ftol=ftol,
               verbose=fit_verbose)

        return LE.embed()
