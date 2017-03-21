import sys
sys.path.append('./src/common/functions')
from external_long_term_memory_func import external_long_term_memory_function
import numpy as np

from chainer import cuda
from chainer import initializers
from chainer import link


class ExternalLongTermMemory(link.Link):
    def __init__(self, n_class, n_units, initial_memory=None, class_memory=50):
        super().__init__()
        self.add_param('memory', (n_class*class_memory, n_units))
        self.add_param('memory_value', (n_class*class_memory, ))  #, dtype=self.xp.int32)
        self.add_param('memory_age', (n_class*class_memory, ))  #, dtype=self.xp.int32)
        self.memory.data.fill(1.0)
        self.memory_value.data[...] = \
            self.xp.repeat(self.xp.arange(n_class), class_memory)

        shuffled_idx = np.random.permutation(n_class*class_memory)
        self.memory_value.data = self.memory_value.data[shuffled_idx]
        self.memory_age.data.fill(0)

        # For backward compatibility
        self.n_class = n_class
        self.n_units = n_units

    def __call__(self, embedding_vecs, memory, value):
        return external_long_term_memory_function(embedding_vecs, memory, value)
