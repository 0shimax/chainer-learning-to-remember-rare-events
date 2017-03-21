import numpy as np
import chainer
from chainer import cuda, function


class ExternalMemoryFunction(function.Function):
    def forward(self, inputs):
        self.xp = cuda.get_array_module(*inputs)

        if len(inputs)==5:
            embedding_vecs, t, self.memory, self.update_weight, train = inputs
        else:
            embedding_vecs, self.memory, self.update_weight, train = inputs
        self.n_class, self.n_unit = self.memory.shape
        weight = self._compute_attention(embedding_vecs)
        if train:
            self._calculate_center(embedding_vecs, t)
        return weight, self.memory,

    def backward(self, inputs, grad_outputs):
        """never backward
        """
        if len(inputs)==5:
            return None,None,None,None,None,
        else:
            return None,None,None,None

    def _normalize(self, vec, vec_sum):
        normalized_vec = \
            self.xp.where( \
                vec==0, \
                self.xp.zeros_like(vec), \
                vec/vec_sum \
            ).transpose(1,0)
        return normalized_vec

    def _calculate_channel_idx(self, embedding_vecs, t):
        represented_vec = self.xp.zeros( \
            (self.n_class, embedding_vecs.shape[1]), dtype=embedding_vecs.dtype)
        for vec, klass_idx in zip(embedding_vecs, t):
            represented_vec[klass_idx] += vec
        return represented_vec

    def _calculate_center(self, embedding_vecs, t):
        n_batch, n_unit, _, _ = embedding_vecs.shape
        vecs = embedding_vecs.reshape((n_batch, n_unit))
        represented_vec = self._calculate_channel_idx(vecs, t)  # (n_class, n_unit)

        represented_vec_sum = represented_vec.sum(axis=1)
        represented_vec = represented_vec.transpose(1,0)

        # normalize
        represented_vec = self._normalize(represented_vec, represented_vec_sum)

        self.memory = \
            (1-self.update_weight)*self.memory \
            + self.update_weight*represented_vec

        external_memory_sum = self.memory.sum(axis=1)
        t_external_memory = self.memory.transpose(1,0)

        # normalize
        self.memory = \
            self._normalize(t_external_memory, external_memory_sum)

    def _compute_attention(self, embedding_vecs):
        '''
        context_vec: (batch_size, n_unit). default (20, 4096).
        '''
        n_batch, n_unit, _, _ = embedding_vecs.shape
        vecs = embedding_vecs.reshape((n_batch, n_unit))
        vecs = self._normalize(vecs.transpose(1,0), vecs.sum(axis=1))
        weights = vecs.dot(self.memory.T)  # (batch_size, n_class)
        weights = self._normalize(weights.transpose(1,0), weights.sum(axis=1))

        return weights
