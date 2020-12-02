import numpy as np
from relegy.metrics import *


def test_rmse():
    n = 50
    similarity_matrix = np.ones((n, n))
    obtained_matrix = np.zeros((n, n))
    assert rmse(similarity_matrix, obtained_matrix) == 1.0


def test_nrmse():
    n = 50
    similarity_matrix = np.ones((n, n))
    obtained_matrix = np.zeros((n, n))
    assert nrmse(similarity_matrix, obtained_matrix) == 1.0


def test_precision_at_k():
    n = 50
    similarity_matrix = np.ones((n, n)) / 2
    obtained_matrix = np.ones((n, n))
    for pk in precision_at_k(similarity_matrix, obtained_matrix):
        assert pk == 1.0


def test_average_precision():
    n = 50
    similarity_vector = np.ones(n) / 2
    obtained_vector = np.ones(n)
    assert average_precision(similarity_vector, obtained_vector) == 1.0


def test_mean_average_precision():
    n = 50
    similarity_matrix = np.ones((n, n)) / 2
    obtained_matrix = np.ones((n, n))
    assert mean_average_precision(similarity_matrix, obtained_matrix) == 1.0


def test_all_average_precision():
    n = 50
    similarity_matrix = np.ones((n, n)) / 2
    obtained_matrix = np.ones((n, n))
    for ap in all_average_precision(similarity_matrix, obtained_matrix):
        assert ap == 1.0
