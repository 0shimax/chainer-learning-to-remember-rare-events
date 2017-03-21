import sys
sys.path.append('./src/common/functions')
import numpy as np
import chainer
from chainer import cuda, function
from func_utils import _get_topk, _normalize


class ExternalLongTermMemoryFunction(function.Function):
    def __init__(self, memory, memory_value):
        self.memory = memory
        self.memory_value = memory_value

    def forward(self, inputs):
        '''
        embedding_vecs: (n_batch, vec_size)
        cos_sim: (n_batch, memory_size)
        '''
        self.xp = cuda.get_array_module(*inputs)

        # embedding_vecs, cos_sim = inputs  # (n_batch, vec_size)
        cos_sim = inputs[0]  # (n_batch, memory_size)
        self.n_menbers, self.n_unit = self.memory.shape  # (memory_size, vec_size)
        # cos_sim = self._calculate_cos_sim(embedding_vecs)  # (n_batch, memory_size)
        self.nbest_v, self.nbest_idx = self._compute_NN_index(cos_sim)

        return self.nbest_idx, self.memory_value.data[self.nbest_idx[:,0]]

    def _compute_NN_index(self, cos_sim_matrix, n_best=256):
        n_batch, memory_size = cos_sim_matrix.shape
        n_best = memory_size if memory_size < n_best else n_best

        return _get_topk(cos_sim_matrix, n_best)


def external_long_term_memory_function(
        x, memory, memory_value):
    return ExternalLongTermMemoryFunction(memory, memory_value)(x)
