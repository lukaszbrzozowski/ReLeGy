import numpy as np
import networkx as nx
from networkx import Graph
from relegy.__base import Model


class LINE(Model):
    """
    The LINE method implementation. \n
    The details may be found in: \n
    ' J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei. Line: Large-scale information network embedding. In
    WWW, 2015.'
    """

    def __init__(self,
                 graph: Graph):
        """
        LINE - constructor (step I)
        @param graph: The graph to be embedded
        """
        super().__init__(graph.to_directed())
        self.__d = None
        self.__lr1 = None
        self.__lr2 = None
        self.__batch_size = None
        self.__lmbd1 = None
        self.__lmbd2 = None
        self.__A = None
        self.__U1 = None
        self.__U2 = None
        self.__Z = None
        self.__E = None
        self.__o1 = None
        self.__o2 = None
        self.__Frob = None
        self.__grad1 = None
        self.__grad2 = None

    @Model._init_in_init_model_fit
    def initialize(self,
                   d: int = 2,
                   **kwargs):
        """
        LINE - initialize (step II) \n
        Generation of loss functions and input matrices.
        @param d: The embedding dimension.
        @param kwargs: Two input matrices 'U1' and 'U2' both of dimension Nxd may be passed here. They are generated
        from the uniform distribution on [0.0, 1.0) otherwise.

        """
        graph = self.get_graph()
        self.__d = d
        self.__E = len(graph.edges)
        self.__U1 = kwargs["U1"] if "U1" in kwargs else np.random.random((len(graph.nodes), self.__d))
        self.__U2 = kwargs["U2"] if "U2" in kwargs else np.random.random((len(graph.nodes), self.__d))
        self.__A = nx.to_numpy_array(graph, nodelist=np.arange(len(graph.nodes)))
        self.__Frob = lambda U: np.linalg.norm(U)
        p1 = lambda x, y: 1 / (1 + np.exp(-np.dot(x, y)))
        p2 =  lambda x, y: np.exp(np.dot(x, y)) / (np.sum(self.__U2 @ y))
        self.__o1 = lambda G: -sum([self.__A[i, j] * np.log(p1(self.__U1[i, :], self.__U1[j, :])) for (i, j) in G.edges])
        self.__o2 = lambda G: -sum([self.__A[i, j] * np.log(p2(self.__U2[i, :], self.__U2[j, :])) for (i, j) in G.edges])
        self.__grad1 = lambda i, j: self.__A[i, j] * (-self.__U1[j, :]) * \
                                    np.exp(-np.dot(self.__U1[i, :], self.__U1[j, :])) / \
                                    (1 + np.exp(-np.dot(self.__U1[i, :], self.__U1[j, :]))) ** 2
        self.__grad2 = lambda i, j: self.__A[i, j] * (
                    self.__U2[j, :] * np.exp(np.dot(self.__U2[i, :], self.__U2[j, :])) * np.sum(
                self.__U2 @ self.__U2[j, :]) - np.exp(np.dot(self.__U2[i, :], self.__U2[j, :])) * self.__U2[j, :]) / (
                                 np.sum(self.__U2 @ self.__U2[j, :])) ** 2

    @Model._init_model_in_init_model_fit
    def initialize_model(self,
                         batch_size: int = 30,
                         lmbd1: float = 1e-1,
                         lmbd2: float = 1e-2,
                         lr1: float = 1e-4,
                         lr2: float = 1e-4):
        """
        LINE - initialize_model (step III) \n
        Sets the learning rate and regularization parameters values.

        @param batch_size: Size of the batch passed during fitting.
        @param lmbd1: Regularization parameter for the first loss function.
        @param lmbd2: Regularization parameter for the second loss function.
        @param lr1: Learning rate for the the first loss function optimization.
        @param lr2: Learning rate for the the second loss function optimization.
        """
        self.__batch_size = batch_size
        self.__lmbd1 = lmbd1
        self.__lmbd2 = lmbd2
        self.__lr1 = lr1
        self.__lr2 = lr2

    @Model._fit_in_init_model_fit
    def fit(self,
            num_iter: int = 400,
            verbose: bool = True):
        """
        LINE - fit (step IV) \n
        Performs the fitting to both loss functions.
        @param num_iter: The number of iterations of the fitting process.
        @param verbose: Verbosity parameter.
        """
        graph = self.get_graph()
        for it in range(num_iter):
            resampled_edges = np.copy(graph.edges)
            np.random.shuffle(resampled_edges)
            mini_batches = [resampled_edges[i * self.__batch_size:min(self.__E, i + 1 * self.__batch_size)] for i in
                            range(self.__E // self.__batch_size)]
            for batch in mini_batches:
                step1 = np.zeros(self.__U1.shape)
                step2 = np.zeros(self.__U2.shape)
                for (i, j) in batch:
                    step1[i, :] += self.__grad1(i, j) + self.__lmbd1 * self.__U1[i, :]
                    step2[i, :] += self.__grad2(i, j) + self.__lmbd2 * self.__U2[i, :]
                self.__U1 -= self.__lr1 * step1
                self.__U2 -= self.__lr2 * step2
            if verbose:
                print(f"Epoch: {it + 1}, "
                      f"1st-order objective: {self.__o1(graph):.2f}, "
                      f"1st-order Frob: {self.__Frob(self.__U1):.2f}, "
                      f"2nd-order objective: {self.__o2(graph):.2f}, "
                      f"2nd-order Frob:{self.__Frob(self.__U2):.2f}",
                      end="\r")

    @Model._embed_in_init_model_fit
    def embed(self):
        """
        LINE - embed (step V) \n
        Returns the embedding as a concatenation of two matrices minimizing the two loss functions.
        @return: The embedding matrix having shape Nx(2*d).
        """
        return np.c_[self.__U1, self.__U2]

    def info(self):
        raise NotImplementedError

    @staticmethod
    def fast_embed(graph: Graph,
                   d: int = 2,
                   batch_size: int = 30,
                   lmbd1: float = 1e-1,
                   lmbd2: float = 1e-2,
                   lr1: float = 1e-4,
                   lr2: float = 1e-4,
                   num_iter: int = 400,
                   fit_verbose: bool = True,
                   **kwargs):
        """
        LINE - fast_embed \n
        Returns the embedding in a single step.

        @param graph: The graph to be embedded. Present in '__init__'
        @param d: The embedding dimension. Present in 'initialize'
        @param batch_size: Size of the batch passed during fitting. Present in 'initialize_model'
        @param lmbd1: Regularization parameter for the first loss function. Present in 'initialize_model'
        @param lmbd2: Regularization parameter for the second loss function. Present in 'initialize_model'
        @param lr1: Learning rate for the the first loss function optimization. Present in 'initialize_model'
        @param lr2: Learning rate for the the second loss function optimization. Present in 'initialize_model'
        @param num_iter: The number of iterations of the fitting process. Present in 'fit'
        @param fit_verbose: Verbosity parameter. Present in 'fit'
        @param kwargs: Two input matrices 'U1' and 'U2' both of dimension Nxd may be passed here. They are generated
        from the uniform distribution on [0.0, 1.0) otherwise. Present in 'initialize'
        @return:
        """
        line = LINE(graph)
        line.initialize(d=d,
                        **kwargs)
        line.initialize_model(batch_size=batch_size,
                              lmbd1=lmbd1,
                              lmbd2=lmbd2,
                              lr1=lr1,
                              lr2=lr2)
        line.fit(num_iter=num_iter,
                 verbose=fit_verbose)
        return line.embed()
