import numpy as np
from collections import namedtuple
import scipy.sparse as sp

SparseMatrix = namedtuple("SparseMatrix", "indices values dense_shape")


def load_karate(path="engthesis/data/karate-club/"):
    """Load karate club dataset"""
    print('Loading karate club dataset...')
    import random

    edges = np.loadtxt("{}edges.txt".format(path), dtype=np.int32) - 1  # 0-based indexing
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]  # reorder list of edges also by second column
    features = sp.eye(np.max(edges + 1), dtype=np.float32).tocsr()

    idx_labels = np.loadtxt("{}mod-based-clusters.txt".format(path), dtype=np.int32)
    labels = idx_labels[idx_labels[:, 0].argsort()]

    labels = np.eye(max(idx_labels[:, 1]) + 1, dtype=np.int32)[idx_labels[:, 1]]  # one-hot encoding of labels

    E = np.concatenate((edges, np.zeros((len(edges), 1), dtype=np.int32)), axis=1)
    N = np.concatenate((features.toarray(), np.zeros((features.shape[0], 1), dtype=np.int32)), axis=1)

    mask_train = np.zeros(shape=(34,), dtype=np.float32)
    mask_test = np.zeros(shape=(34,), dtype=np.float32)
    idx_classes = np.argmax(labels, axis=1)
    #
    # id_0, id_4, id_5, id_12 = random.choices(np.argwhere(idx_classes == 0), k=4)
    # id_1, id_6, id_7, id_13 = random.choices(np.argwhere(idx_classes == 1), k=4)
    # id_2, id_8, id_9, id_14 = random.choices(np.argwhere(idx_classes == 2), k=4)
    # id_3, id_10, id_11, id_15 = random.choices(np.argwhere(idx_classes == 3), k=4)
    #
    # mask_train[id_0] = 1.  # class 1
    # mask_train[id_1] = 1.  # class 2
    # mask_train[id_2] = 1.  # class 0
    # mask_train[id_3] = 1.  # class 3
    # mask_test = 1. - mask_train

    return E, N, idx_labels, mask_train, mask_test