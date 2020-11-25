import numpy as np
from networkx import Graph
import networkx as nx
from numpy import ndarray

from relegy.__helpers.sdae import SDAE
from relegy.__base import Model

init_verification = {"T": [(lambda x: x > 0, "'T' must be greater than 0.")],
                     "alpha": [(lambda x: 0 <= x <= 1, "'alpha' must be in range [0, 1].")]}

init_model_verification = {"d": [(lambda x: x > 0, "'d' must be greater than 0.")],
                           "n_layers": [(lambda x: x > 0, "'n_layers' must be greater than 0.")],
                           "n_hid": [(lambda x: True if x is None else np.all(x), "Every element of 'n_hid' must be greater than 0.")],
                           "dropout": [(lambda x: np.all(x) >= 0, "Every element of 'dropout' must be non-negative")],
                           "bias": [(lambda x: (x == 0) or (x == 1), "'bias' must be greater than boolean or either 0 or 1")],
                           "batch_size": [(lambda x: x > 0, "'batch_size' must be greater than 0")]}

fit_verification = {"num_iter": [(lambda x: x > 0, "num_iter must be greater than 0")]}

fast_embed_verification = Model.dict_union(init_verification, init_model_verification, fit_verification)


class DNGR(Model):
    """
    The DNGR method implementation. \n
    The details may be found in: \n
    'S. Cao, W. Lu, and Q. Xu. Deep neural networks for learning graph representations. In AAAI, 2016.'
    """
    def __init__(self,
                 graph: Graph):
        """
        DNGR - constructor (step I)

        @param graph: The graph to be embedded. Nodes of the graph must be a sorted array from 0 to n-1, where n is
        the number of vertices.
        """

        super().__init__(graph)
        self.__d = None
        self.__alpha = None
        self.__T = None
        self.__model = None
        self.__ppmi = None
        self.__n_layers = None
        self.__n_hid = None
        self.__dropout = None
        self.__enc_act = None
        self.__dec_act = None
        self.__bias = None
        self.__loss_fn = None
        self.__batch_size = None
        self.__optimizer = None

    @Model._init_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_verification)
    def initialize(self,
                   alpha: float = 0.9,
                   T: int = 40):
        """
        DNGR - initialize (step II) \n
        Calculates the PPMI matrix.

        @param alpha: probability, that the random surfing will continue instead of coming back to the first vertex.
        @param T: The length of a single random walk.
        """
        self.__alpha = alpha
        self.__T = T

        rs = self.__random_surf()
        self.__ppmi = self.__get_ppmi(rs)

    @Model._init_model_in_init_model_fit
    @Model._verify_parameters(rules_dict=init_model_verification)
    def initialize_model(self,
                         d: int = 2,
                         n_layers: int = 1,
                         n_hid=None,
                         dropout: float = 0.05,
                         enc_act="sigmoid",
                         dec_act="linear",
                         bias: bool = True,
                         loss_fn: str = "mse",
                         batch_size: int = 32,
                         optimizer: str = "adam"):
        """
        DNGR - initialize_model (step III) \n
        Sets values of the given parameters.

        @param d: The embedding dimension.
        @param n_layers: Number of layers of the Stacked Denoising Auto-Encoder.
        @param n_hid: List of number of vertices in each layer.
        @param dropout: Percentage of dropout vertices in each layer.
        @param enc_act: Name/names of activation functions in the encoding layers.
        @param dec_act: Name/names of activation functions in the decoding layers.
        @param bias: Boolean value, if True, bias is present in the network.
        @param loss_fn: Name of the loss function for the network.
        @param batch_size: Batch size for the network.
        @param optimizer: Name of the optimizer used in network learning.
        """
        self.__d = d
        if n_hid is None:
            n_hid = d
        elif type(n_hid) == int:
            assert (n_hid == d)
        else:
            assert (d == n_hid[-1])

        self.__n_layers = n_layers
        self.__n_hid = n_hid
        self.__dropout = dropout
        self.__enc_act = enc_act
        self.__dec_act = dec_act
        self.__bias = bias
        self.__loss_fn = loss_fn
        self.__batch_size = batch_size
        self.__optimizer = optimizer

    def __random_surf(self) -> ndarray:
        A = nx.to_numpy_array(self.get_graph(), nodelist=np.arange(len(self.get_graph().nodes)))
        scaled_A = self.__scale_sim_mat(A)
        P0 = np.identity(A.shape[0])
        P = P0
        M = np.zeros(A.shape)

        for i in range(self.__T):
            P = self.__alpha * (P @ scaled_A) + (1 - self.__alpha) * P0
            M += P
        return M

    @staticmethod
    def __scale_sim_mat(A):
        W = A - np.diag(np.diag(A))
        D = np.diag(1 / np.sum(A, axis=0))
        scaled_A = D @ W
        return scaled_A

    def __get_ppmi(self, M):
        scaled_M = self.__scale_sim_mat(M)
        N = scaled_M.shape[0]
        colsum = np.sum(scaled_M, axis=0).reshape(1, N)
        rowsum = np.sum(scaled_M, axis=1).reshape(N, 1)
        allsum = np.sum(colsum)
        PPMI = np.log((allsum * scaled_M) / np.dot(rowsum, colsum))
        PPMI[np.isinf(PPMI)] = 0
        PPMI[PPMI < 0] = 0
        return PPMI

    @Model._fit_in_init_model_fit
    @Model._verify_parameters(rules_dict=fit_verification)
    def fit(self,
            num_iter: int = 300,
            verbose: bool = True,
            random_state: int = None):
        """
        DNGR - fit (step IV) \n
        Train the SDAE network layer by layer.

        @param num_iter: Number of iterations.
        @param verbose: Verbosity parameter.
        @param random_state: Initial random state for training.
        """

        if random_state is not None:
            np.random.seed(random_state)

        sdae = SDAE(self.__n_layers,
                    self.__n_hid,
                    self.__dropout,
                    self.__enc_act,
                    self.__dec_act,
                    self.__bias,
                    self.__loss_fn,
                    self.__batch_size,
                    num_iter,
                    self.__optimizer,
                    verbose)

        final_model, data_in, mse = sdae.get_pretrained_sda(self.__ppmi, True, False)
        self.__model = final_model
        self.__model.compile(self.__optimizer, self.__loss_fn)

    @Model._embed_in_init_model_fit
    def embed(self) -> ndarray:
        """
        DNGR - embed (step V) \n
        Returns the embedding from the SDAE network.

        @return: The embedding matrix.
        """
        Z = self.__model.predict(self.__ppmi)
        return Z

    @staticmethod
    @Model._verify_parameters(rules_dict=fast_embed_verification)
    def fast_embed(graph: Graph,
                   alpha: float = 0.9,
                   T: int = 40,
                   d: int = 2,
                   n_layers: int = 1,
                   n_hid=None,
                   dropout: float = 0.05,
                   enc_act="sigmoid",
                   dec_act="linear",
                   bias: bool = True,
                   loss_fn: str = "mse",
                   batch_size: int = 32,
                   optimizer: str = "adam",
                   num_iter: int = 300,
                   fit_verbose: bool = True,
                   random_state: int = None
                   ) -> ndarray:
        """
        DNGR - fast_embed \n
        Returns the embedding in a single step.

        @param graph: The graph to be embedded. Present in '__init__'
        @param alpha: probability, that the random surfing will continue instead of coming back to the first vertex. Present in 'initialize'
        @param T: The length of a single random walk. Present in 'initialize'
        @param d: The embedding dimension. Present in 'initialize_model'
        @param n_layers: Number of layers of the Stacked Denoising Auto-Encoder. Present in 'initialize_model'
        @param n_hid: List of number of vertices in each layer. Present in 'initialize_model'
        @param dropout: Percentage of dropout vertices in each layer. Present in 'initialize_model'
        @param enc_act: Name/names of activation functions in the encoding layers. Present in 'initialize_model'
        @param dec_act: Name/names of activation functions in the decoding layers. Present in 'initialize_model'
        @param bias: Boolean value, if True, bias is present in the network. Present in 'initialize_model'
        @param loss_fn: Name of the loss function for the network. Present in 'initialize_model'
        @param batch_size: Batch size for the network. Present in 'initialize_model'
        @param optimizer: Name of the optimizer used in network learning. Present in 'initialize_model'
        @param num_iter: Number of iterations. Present in 'fit'
        @param fit_verbose: Verbosity parameter. Present in 'fit'
        @param random_state: Initial random state for training. Present in 'fit'
        """
        dngr = DNGR(graph)
        dngr.initialize(alpha=alpha,
                        T=T)
        dngr.initialize_model(d=d,
                              n_layers=n_layers,
                              n_hid=n_hid,
                              dropout=dropout,
                              enc_act=enc_act,
                              dec_act=dec_act,
                              bias=bias,
                              loss_fn=loss_fn,
                              batch_size=batch_size,
                              optimizer=optimizer)
        dngr.fit(num_iter=num_iter,
                 verbose=fit_verbose,
                 random_state=random_state)
        return dngr.embed()
