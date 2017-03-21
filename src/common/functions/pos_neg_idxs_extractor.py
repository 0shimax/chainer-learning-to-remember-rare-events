import sys
sys.path.append('./src/common/functions')
import numpy
import random

import chainer
from chainer import cuda
from chainer import function
from chainer.functions.activation import log_softmax
from chainer.utils import type_check

from func_utils import _get_topk, _normalize


class PosNegIdxsExtractor(function.Function):
    # mergine
    alpha = 0.1

    def __init__(self, memory, memory_value, hint_pool_idxs):
        self.memory = memory
        self.memory_value = memory_value
        self.hint_pool_idxs = hint_pool_idxs

    def forward(self, inputs):
        self.xp = cuda.get_array_module(*inputs)

        # all_similarities: (n_batch, memory_size)
        # hint_pool_idxs: (n_batch, n_best)
        all_similarities, gt = inputs
        n_batch, n_best = self.hint_pool_idxs.shape
        memory_size, vec_size = self.memory.shape

        # (n_batch, n_best, vec_size)
        gathered_mem_keys = \
            self.memory[self.hint_pool_idxs.ravel()].reshape((n_batch, n_best, vec_size))  # my_mem_keys

        # all_similarities: (n_batch, memory_size)
        # hint_pool_sims: (n_batch, n_best)
        hint_pool_sims = all_similarities[:, self.hint_pool_idxs][:,0]

        # hint_pool_mem_vals: (n_batch, n_best)
        hint_pool_mem_vals = \
            self.memory_value[self.hint_pool_idxs.ravel()].reshape((n_batch, n_best))

        # teacher_idxs, self.incorrect_memory_lookup, infered_class = \
        #     self._extract_idxs(gt, hint_pool_mem_vals, hint_pool_sims)

        teacher_idxs, self.incorrect_memory_lookup = \
            self._extract_idxs(gt, hint_pool_mem_vals, hint_pool_sims)

        return teacher_idxs, self.neg_idxs, self.incorrect_memory_lookup

    def _extract_idxs(self, gt, hint_pool_mem_vals, hint_pool_sims):
        '''
        hint_pool_mem_vals:(n_batch, n_best)
        hint_pool_sims:(n_batch, n_best)
        '''
        n_batch, n_best = hint_pool_mem_vals.shape
        # prepare hints from the teacher on hint pool
        # 0 if correct else 1
        # teacher_hints: (n_batch, n_best)
        teacher_hints = self.xp.abs( \
            self.xp.broadcast_to(gt, (n_best, n_batch)).T \
            - hint_pool_mem_vals)

        # tf.minimum(1.0, teacher_hints): 0 if correct else 1
        # output: 1 if correct else 0
        teacher_hints = 1.0 - self.xp.minimum(1.0, teacher_hints)
        teacher_vals, teacher_hint_idxs = _get_topk(
            hint_pool_sims * teacher_hints, k=1)
        neg_teacher_vals, neg_hint_idxs = _get_topk(
            hint_pool_sims * (1 - teacher_hints), k=1)

        # bring back idxs to full memory
        # teacher_idxs is top1 of self.hint_pool_idxs.
        # teacher_idxs(correctly infered idxs) is top1 idx per batch in memory_size.
        # memory[self.hint_pool_idxs[teacher_hint_idxs]]
        # extract top1 idx from hint_pool_idx
        correct_memory_idx = teacher_hint_idxs[:, 0] \
                                + n_best * self.xp.arange(n_batch)
        # memory idxs
        teacher_idxs = \
            self.hint_pool_idxs.reshape(self.hint_pool_idxs.size)[correct_memory_idx]

        in_correct_memory_idx = neg_hint_idxs[:, 0] \
                                + n_best * self.xp.arange(n_batch)
        # memory idxs
        self.neg_idxs = \
            self.hint_pool_idxs.reshape(self.hint_pool_idxs.size)[in_correct_memory_idx]
        self.teacher_idxs = teacher_idxs.copy()

        # zero-out teacher_vals if there are no hints(there are no correct label)
        # teacher_hints: (n_batch, n_best)
        # teacher_vals: (n_batch, 1)
        teacher_vals *= \
            (1.0 - self.xp.where(teacher_hints.sum(axis=1)==0, 1, 0))[:,self.xp.newaxis]

        # we'll determine whether to do an update to memory based on whether
        # memory was queried correctly
        # incorrect_memory_lookup: (n_batch,)
        incorrect_memory_lookup = self.xp.where(teacher_hints.sum(axis=1)==0, 1, 0)

        '''
        # loss based on triplet loss
        teacher_loss = \
            self.xp.maximum(0, neg_teacher_vals - teacher_vals + self.alpha) \
            - self.alpha
        '''
        # infer
        # infered_class = self._predict_label(hint_pool_sims, n_batch, n_best)

        return teacher_idxs, incorrect_memory_lookup
        # return teacher_idxs, incorrect_memory_lookup, infered_class

    def _predict_label(hint_pool_sims, n_batch, n_best):
        nearest_neighbor = self.xp.argmax(hint_pool_sims[:, :n_batch], 1)

        no_teacher_memory_idx = nearest_neighbor + n_best * self.xp.arange(n_batch)
        no_teacher_idxs = \
            self.hint_pool_idxs.reshape(self.hint_pool_idxs.size)[no_teacher_memory_idx]
        return self.memory_value[no_teacher_idxs]


def pos_neg_idxs_extractor(memory, memory_value, hint_pool_idxs, x, t):
    return PosNegIdxsExtractor(memory, memory_value, hint_pool_idxs)(x, t)
