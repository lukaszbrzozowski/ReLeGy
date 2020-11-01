import numpy as np
from networkx import to_numpy_array, Graph
from numpy import ndarray

from engthesis.helpers.sdae import SDAE
from engthesis.model import Model


class DNGR(Model):

    def __init__(self,
                 graph: Graph,):

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
    def initialize(self,
                   alpha: float = 0.9,
                   T: int = 40):
        self.__alpha = alpha
        self.__T = T

        rs = self.__random_surf()
        self.__ppmi = self.__get_ppmi(rs)

    @Model._init_model_in_init_model_fit
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

    def info(self) -> str:
        raise NotImplementedError

    def __random_surf(self) -> ndarray:
        A = to_numpy_array(self.get_graph(), nodelist=np.arange(len(self.get_graph().nodes)))
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
    def fit(self,
            num_iter: int = 300,
            verbose: bool = True,
            random_state: int = None):

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
        Z = self.__model.predict(self.__ppmi)
        return Z

    @staticmethod
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
