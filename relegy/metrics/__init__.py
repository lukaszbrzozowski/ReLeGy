import numpy as np


def rmse(similarity_matrix, obtained_matrix):
    """
    Rooted mean square error of the representation.
    :param similarity_matrix: The approximated matrix, usually the adjacency matrix of a graph
    :param obtained_matrix: The matrix obtained with a representation method
    :return: The value of RMSE
    """
    N = similarity_matrix.shape[0]
    frob_error_squared = np.linalg.norm(similarity_matrix - obtained_matrix)
    return frob_error_squared / N


def nrmse(similarity_matrix, obtained_matrix):
    """
    Normalised mean square error of the representation.
    :param similarity_matrix: The approximated matrix, usually the adjacency matrix of a graph
    :param obtained_matrix: The matrix obtained with a representation method
    :return: The value of NRMSE
    """
    frob_error = np.linalg.norm(similarity_matrix - obtained_matrix)
    similarity_matrix_norm = np.linalg.norm(similarity_matrix)
    return frob_error / similarity_matrix_norm


def precision_at_k(similarity_matrix, obtained_matrix):
    """
    Precision at k for each k in range 0 to the number of non-zero elements of obtained_matrix
    :param similarity_matrix: The approximated matrix, usually the adjacency matrix of a graph
    :param obtained_matrix: The matrix obtained with a representation method
    :return: The value of NRMSE
    """
    flat_sim = similarity_matrix.flatten()
    flat_obs = obtained_matrix.flatten()
    ars = np.argsort(flat_obs)[::-1].flatten()
    sorted_flat_obs = flat_obs[ars]
    sorted_flat_sim = flat_sim[ars]
    if len(sorted_flat_obs > 0) == 0:
        if np.sum(flat_sim) == 0:
            return np.repeat(1, flat_sim.shape[0])
        else:
            return np.repeat(0, flat_sim.shape[0])
    ep = np.cumsum(sorted_flat_obs > 0)
    true_predictions = sorted_flat_sim > 0
    return np.cumsum(true_predictions) / ep


def average_precision(similarity_vector, obtained_vector):
    """
    Average precision of prediction
    :param similarity_vector: The approximated vector, usually a column from the adjacency matrix of a graph
    :param obtained_vector: The vector obtained with a representation method
    :return: The value of NRMSE
    """
    prec_k = precision_at_k(similarity_vector, obtained_vector)
    flat_sim_positive = similarity_vector > 0

    return np.sum(prec_k * flat_sim_positive) / np.sum(flat_sim_positive)


def mean_average_precision(similarity_matrix, obtained_matrix):
    """
    Mean average precision of prediction. Mean precision is calculated for each vertex of a graph.
    :param similarity_matrix: The approximated matrix, usually the adjacency matrix of a graph
    :param obtained_matrix: The matrix obtained with a representation method
    :return: The value of NRMSE
    """
    return np.mean([average_precision(similarity_matrix[i, :], obtained_matrix[i, :])
                    for i in range(similarity_matrix.shape[0])])


def all_average_precision(similarity_matrix, obtained_matrix):
    """
    A vector of average precision values for each vertex.
    :param similarity_matrix: The approximated matrix, usually the adjacency matrix of a graph
    :param obtained_matrix: The matrix obtained with a representation method
    :return: The value of NRMSE
    """
    return np.array([average_precision(similarity_matrix[i, :], obtained_matrix[i, :])
                     for i in range(similarity_matrix.shape[0])])
