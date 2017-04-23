import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import sys
sys.path.append('./src/common/loss_functions')
sys.path.append('./src/common/linker')
sys.path.append('./src/common/functions')
from memory_loss import memory_loss
from pos_neg_idxs_extractor import pos_neg_idxs_extractor
from external_long_term_memory import ExternalLongTermMemory
from update_memory import update_memory
from external_memory_accuracy import external_memory_accuracy


class Fire(chainer.Chain):
    def __init__(self, in_size, s1, e1, e3):
        super().__init__(
            conv1=L.Convolution2D(in_size, s1, 1),
            conv2=L.Convolution2D(s1, e1, 1),
            conv3=L.Convolution2D(s1, e3, 3, pad=1),
        )

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h_1 = self.conv2(h)
        h_3 = self.conv3(h)
        h_out = F.concat([h_1, h_3], axis=1)
        return F.elu(h_out)


class Encoder(chainer.Chain):
    def __init__(self, in_ch, vec_size=128):
        super().__init__(
            conv1=L.Convolution2D(in_ch, 96, 7, stride=2, pad=3),
            fire2=Fire(96, 16, 64, 64),
            fire3=Fire(128, 16, 64, 64),
            fire4=Fire(128, 16, 128, 128),
            # fire5=Fire(256, 32, 128, 128),
            # fire6=Fire(256, 48, 192, 192),
            # fire7=Fire(384, 48, 192, 192),
            # fire8=Fire(384, 64, 256, 256),
            # fire9=Fire(512, 64, 256, 256),

            conv9=L.Convolution2D(256*21, vec_size, 1, pad=0),  # *21
        )

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire2(h)
        h = self.fire3(h)
        h = self.fire4(h)

        # h = F.max_pooling_2d(h, 3, stride=2)
        #
        # h = self.fire5(h)
        # h = self.fire6(h)
        # h = self.fire7(h)
        # h = self.fire8(h)

        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.elu(self.conv9(h))

        query = F.reshape(h, h.data.shape[:2])
        return query


class SqueezeNetExternalLongTermMemory(chainer.Chain):
    version = '0.1.0'
    alpha = 0.1
    def __init__(self, n_class, in_ch, update_weight=0.5, vec_size=128):
        super().__init__(
            encoder=Encoder(in_ch, vec_size),
            apply_memory=ExternalLongTermMemory(n_class, vec_size),
        )

        self.train = True
        self.n_class = n_class
        self.active_learn = False
        self.update_weight = update_weight
        self.memory = None
        self.memory_value = None
        self.memory_age = None

    def clear(self):
        self.loss = 0
        self.accuracy = None
        if self.memory is None:
            self.memory = chainer.Variable(self.apply_memory.memory.data)
            self.memory_value = chainer.Variable(self.apply_memory.memory_value.data)
            self.memory_age = chainer.Variable(self.apply_memory.memory_age.data)

    def __call__(self, x, t=None):
        self.clear()
        x = self.xp.reshape(x, (-1, 1, 28, 28))
        x = chainer.Variable(x)
        t = chainer.Variable(t)

        # print("init memory")
        # print(self.memory_value.data)
        if self.train:
            self.apply_memory.memory.data = self.memory.data.copy()
            self.apply_memory.memory_value.data = self.memory_value.data.copy()
            self.apply_memory.memory_age.data = self.memory_age.data.copy()

        query = self.encoder(x)

        # using external memory from here
        normed_query = F.normalize(query)

        self.memory = chainer.Variable( \
            self.apply_memory.memory.data)
        self.memory_value = chainer.Variable( \
            self.apply_memory.memory_value.data)
        self.memory_age = chainer.Variable( \
            self.apply_memory.memory_age.data)

        all_similarities = F.matmul( \
            normed_query, self.memory, transb=True)  # (n_batch, memory_size)

        with chainer.no_backprop_mode():
            # hint_pool_idxs: (n_batch, n_best)
            # infered_class: (n_batch)
            hint_pool_idxs, infered_class = \
                self.apply_memory(all_similarities, self.memory, self.memory_value)

            teacher_idxs, neg_idxs, incorrect_memory_lookup = \
                pos_neg_idxs_extractor(self.memory.data,
                            self.memory_value.data,
                            hint_pool_idxs.data,
                            all_similarities,
                            t)

        n_batch = len(t.data)
        teacher_vals = F.select_item(all_similarities, teacher_idxs)
        neg_teacher_vals = F.select_item(all_similarities, neg_idxs)

        self.loss = \
            F.relu(neg_teacher_vals - teacher_vals + self.alpha) \
            - self.alpha
        self.loss = F.sum(self.loss)
        self.loss /= n_batch

        if self.train:
            with chainer.no_backprop_mode():
                self.memory, self.memory_value, self.memory_age = \
                    update_memory( \
                        normed_query, teacher_idxs, incorrect_memory_lookup,
                        t, self.memory.data, \
                        self.memory_value.data, self.memory_age.data)
                self.memory = F.normalize(self.memory)
                self.apply_memory.memory.data = self.memory.data.copy()
                self.apply_memory.memory_value.data = self.memory_value.data.copy()
                self.apply_memory.memory_age.data = self.memory_age.data.copy()

        with chainer.no_backprop_mode():
            self.accuracy = external_memory_accuracy(infered_class, t)
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
