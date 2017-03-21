import random
import numpy as np
import chainer
from chainer import cuda, function
from func_utils import _get_topk, _normalize, _get_topk_for_vec


class UpdateMemory(function.Function):
    def __init__(self, memory, memory_value, memory_age, age_noise):
        self.memory = memory
        self.memory_value = memory_value
        self.memory_age = memory_age
        self.age_noise = age_noise

    def forward(self, inputs):
        self.xp = cuda.get_array_module(*inputs)

        normalized_query, teacher_idxs, incorrect_memory_lookup, gt = inputs
        self._update(normalized_query, teacher_idxs, incorrect_memory_lookup, gt)
        return self.memory, self.memory_value, self.memory_age,

    def _update(self, \
                normalized_query, teacher_idxs, incorrect_memory_lookup, gt):
        # prepare memory updates
        update_keys = normalized_query.copy()
        update_vals = gt.copy()  # intended_output is gt

        fetched_idxs, fetched_keys, fetched_vals, memory_age_with_noise = \
            self._prepare_update( \
                update_keys, update_vals, normalized_query, gt, teacher_idxs)
        self._update_core( \
            update_keys, update_vals, memory_age_with_noise, \
            incorrect_memory_lookup, fetched_idxs, fetched_keys, fetched_vals, gt)

    def _prepare_update(self, update_keys, update_vals, \
                                    normalized_query, gt, teacher_idxs):
        # correct index number in memory(correctly fetched from memory)
        # fetched_idxs: (n_batch, )
        fetched_idxs = teacher_idxs.copy()
        # fetched_keys: (n_batch, vec_size)
        fetched_keys = self.memory[fetched_idxs]  # (memory_size, vec_size)
        # fetched_vals: (n_batch, )
        fetched_vals = self.memory_value[fetched_idxs]

        # do memory updates here
        fetched_keys_upd = update_keys + fetched_keys  # Momentum-like update

        # #TODO: output fetched_keys_upd to normalize
        # fetched_keys_upd = tf.nn.l2_normalize(fetched_keys_upd, dim=1)

        memory_size, vec_size = self.memory.shape
        # Randomize age a bit, e.g., to select different ones in parallel workers.
        memory_age_with_noise = \
            self.memory_age \
            + self.xp.random.randint( \
                - self.age_noise, high=self.age_noise, size=memory_size)

        return fetched_idxs, fetched_keys_upd, fetched_vals, memory_age_with_noise

    def _update_core(self, update_keys, update_vals, memory_age_with_noise, \
                    incorrect_memory_lookup, fetched_idxs, fetched_keys, \
                    fetched_vals, intended_output):
        '''
        incorrect_memory_lookup: incorrect infered flag. (n_batch,)
        oldest_idxs: memory indexs that memory age is old.(n_batch, )
        fetched_idxs: teacher_idxs. correct indexs. (n_batch, )

        fetched_keys: keyes of correctly infered.

        update_vals: gt. (n_batch, )
        update_vals: label of correctly infred. (n_batch, )
        '''
        n_batch, vec_size = fetched_keys.shape
        _, oldest_idxs = _get_topk_for_vec(memory_age_with_noise, k=n_batch)
        # upd_idxs: (n_batch, )
        # if incorrect take oldest_idx, else take correct idx.
        upd_idxs = self.xp.where(incorrect_memory_lookup,  # (n_batch, )
                                  oldest_idxs,
                                  fetched_idxs)
        # update_keys: (n_batch, vec_size)
        # fetched_keys: (n_batch, vec_size)
        upd_keys = self.xp.where(
            self.xp.broadcast_to(
                incorrect_memory_lookup, (vec_size, n_batch)).T,
                update_keys,
                fetched_keys)
        upd_vals = self.xp.where(incorrect_memory_lookup,
                                update_vals,
                                fetched_vals)
        self._make_update(upd_idxs, upd_keys, upd_vals, intended_output)
        # update_op = make_update_op

    def _make_update(self, upd_idxs, upd_keys, \
                                upd_vals, gt, use_recent_idx=False):
        """Function that creates all the update ops."""
        # intended_outpu is gt.

        self.memory_age += 1

        self.memory_age[upd_idxs] = 0
        self.memory[upd_idxs] = upd_keys
        self.memory_value[upd_idxs] = upd_vals

        if use_recent_idx:
            recent_idx_upd = self.recent_idx.copy()
            recent_idx_upd[upd_idxs] = gt
        else:
            recent_idx_upd = None


def update_memory(normalized_query, teacher_idxs, incorrect_memory_lookup,
                    gt, memory, memory_value, memory_age
                    , age_noise=8):
    return UpdateMemory(memory, memory_value, memory_age, age_noise)( \
                    normalized_query, teacher_idxs, incorrect_memory_lookup,gt)
