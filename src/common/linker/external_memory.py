import sys
sys.path.append('./src/common/functions')
from external_memory_func import ExternalMemoryFunction

from chainer import cuda
from chainer import initializers
from chainer import link


class ExternalMemory(link.Link):
    def __init__(self, n_class, n_units, initial_memory=None):
        super().__init__()
        self.add_param('memory', (n_class, n_units))
        self.memory.data.fill(0.0)

        # For backward compatibility
        self.n_class = n_class
        self.n_units = n_units

    def _initialize_params(self):
        self.add_param('memory', (self.n_class, self.n_units))
        self.memory.data.fill(0.0)

    def __call__(self, embedding_vecs, t=None, update_weight=0.5, train=True):
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params()
        self.update_weight = self.xp.array(update_weight, dtype=self.xp.float32)
        if t is None:
            train = self.xp.array(False)
            return ExternalMemoryFunction()( \
                embedding_vecs, self.memory, self.update_weight, train)
        else:
            train = self.xp.array(train)
            return ExternalMemoryFunction()( \
                embedding_vecs, t, self.memory, self.update_weight, train)
