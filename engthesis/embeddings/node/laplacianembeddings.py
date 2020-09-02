from typing import Dict, Any, Union, Callable

from networkx import laplacian_matrix, to_numpy_matrix, Graph
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from engthesis.model.base import Model
import numpy as np


class LaplacianEmbeddings(Model):

    def __init__(self,
                 graph: Graph,
                 d: int = 2):
        """
        The initialization method of the Laplacian Embeddings model.
        :param graph: The graph to be embedded
        :param d: dimensionality of the embedding vectors
        :param similarity_matrix: Similarity matrix of the graph. Adjacency matrix of the graph is passed by default
        """
        super().__init__(graph)

        self.__A: csr_matrix = to_numpy_matrix(self.get_graph(), nodelist=np.arange(graph.nodes))
        self.__d = d


    def info(self) -> str:
        return "To be implemented"

    def embed(self,
              ftol: float = 1e-7,
              verbose: bool = True,
              maxiter: int = 200) -> np.ndarray:
        """
        The embedding method of the Laplacian Eigenmaps.
        :param ftol: Precision parameter of the optimisation process. Default 1e-7
        :param verbose: Whether to print optimisation results after the embedding
        :param maxiter: Maximal number of iterations of the optimisation process
        :return: The graph embedding in R^d
        """

        n = len(self.get_graph().nodes)

        L = laplacian_matrix(self.get_graph(), nodelist=np.arange(n))
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
                       options={'ftol': ftol, 'disp': verbose, 'maxiter': maxiter})
        return res.x.reshape(-1, self.__d)
