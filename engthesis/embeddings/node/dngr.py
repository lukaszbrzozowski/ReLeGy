from engthesis.model.base import Model
from networkx import to_numpy_array
import numpy as np
from numpy import ndarray
from engthesis.helpers.sdae import SDAE


class DNGR(Model):

    def __init__(self, graph, d = 2, alpha = 0.9, T = 10):
        """

        :param graph: Graph to be embedded
        :param kwargs: set of parameters of the method:
        - d - dimension of the embedding
        - alpha - probability of continuing the random walk in the random surfing
        - T - the length of random walks
        - gamma - number of random walks starting from each vertex

        """
        super().__init__(graph)
        self.__d: int = d
        self.__alpha: float = alpha
        self.__T: int = T
        self.__model = None

    def info(self) -> str:
        return "TBI"

    def __random_surf(self) -> ndarray:
        A = to_numpy_array(self.get_graph())
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

    def embed(self, n_layers=1, n_hid=500, dropout=0.05, enc_act='sigmoid', dec_act='linear', bias=True,
              loss_fn='mse', batch_size=32, nb_epoch=300, optimizer='adam', verbose=1, get_enc_model=True,
              get_enc_dec_model=False) -> ndarray:
        assert(self.__d == n_hid[-1])
        M = self.__random_surf()
        PPMI = self.__get_ppmi(M)
        sdae = SDAE(n_layers, n_hid, dropout, enc_act, dec_act, bias, loss_fn, batch_size, nb_epoch, optimizer, verbose)
        final_model, data_in, mse = sdae.get_pretrained_sda(PPMI, get_enc_model, get_enc_dec_model)
        self.__model = final_model
        self.__model.compile(optimizer, loss_fn)
        Z = self.__model.predict(PPMI)

        return Z
