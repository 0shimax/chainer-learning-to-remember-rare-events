import numpy as np
import chainer
from chainer import cuda, function

def _get_topk_for_vec(in_x, k=256):
    # axis is fixed 1.
    vec_size = in_x.shape[0]
    xp = cuda.get_array_module(in_x)
    ids_matrix = xp.empty(k, dtype=xp.int32)
    scores_matrix = xp.empty(k, dtype=xp.float32)
    x = in_x.copy()
    for i in range(k):
        ids = xp.argmax(x).astype('i')  # (n_batch, )
        # if axis == 0:
        #     scores = x[ids]
        #     x[ids] = - float('inf')
        # else:
        ids_matrix[i] = ids
        scores_matrix[i] = x[ids]
        x[ids] = - float('inf')
    return scores_matrix, ids_matrix,


def _get_topk(in_x, k=256, axis=1):
    # axis is fixed 1.
    n_batch, memory_size = in_x.shape
    xp = cuda.get_array_module(in_x)
    ids_matrix = xp.empty((n_batch, k), dtype=xp.int32)
    scores_matrix = xp.empty((n_batch, k), dtype=xp.float32)
    x = in_x.copy()
    for i in range(k):
        ids = xp.argmax(x, axis=axis).astype('i')  # (n_batch, )
        # if axis == 0:
        #     scores = x[ids]
        #     x[ids] = - float('inf')
        # else:
        ids_matrix[:, i] = ids
        scores_matrix[:, i] = x[xp.arange(ids.shape[0]), ids]
        x[xp.arange(ids.shape[0]), ids] = - float('inf')
    return scores_matrix, ids_matrix,


def _normalize(matrix, axis=1):
    xp = cuda.get_array_module(matrix)

    vec = matrix.transpose(1,0)
    vec_sum = matrix.sum(axis=axis)

    normalized_vec = \
        xp.where( \
            vec==0, \
            xp.zeros_like(vec, dtype=vec.dtype), \
            vec/vec_sum \
        ).transpose(1,0)
    return normalized_vec
