import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import sys
sys.path.append('./src/common/loss_functions')
sys.path.append('./src/common/active_learning_tools')
sys.path.append('./src/common/linker')
from median_weighted_softmax_cross_entropy import median_weighted_softmax_cross_entropy
from al_operator_in_net import mask_gt_for_active_learning
from memory_loss import memory_loss
from external_memory import ExternalMemory
import subprocess


def exists_gpu():
    cmd ='lspci | grep VGA | cut -d" " -f 1'
    process = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    b_out, b_err = process.communicate()
    return len(b_out)>0


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


class SqueezeNetExternalMemory(chainer.Chain):
    version = '0.1.0'
    def __init__(self, n_class, in_ch, update_weight=0.5, memory_size=4096):
        super().__init__(
            conv1=L.Convolution2D(in_ch, 96, 7, stride=2, pad=3),
            fire2=Fire(96, 16, 64, 64),
            fire3=Fire(128, 16, 64, 64),
            fire4=Fire(128, 16, 128, 128),
            fire5=Fire(256, 32, 128, 128),
            fire6=Fire(256, 48, 192, 192),
            fire7=Fire(384, 48, 192, 192),
            fire8=Fire(384, 64, 256, 256),
            fire9=Fire(512, 64, 256, 256),

            conv9=L.Convolution2D(512*21, memory_size, 1, pad=0),  # *21
            conv_infer=L.Convolution2D(memory_size, n_class, 1, pad=0),
            apply_memory=ExternalMemory( \
                n_class, memory_size),
        )

        self.train = True
        self.n_class = n_class
        self.active_learn = False
        self.update_weight = update_weight

    def clear(self):
        self.loss = 0
        self.accuracy = None

    def __call__(self, x, t=None):
        self.clear()
        x.volatile = not self.train
        t.volatile = 'AUTO'

        h = F.elu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire2(h)
        h = self.fire3(h)
        h = self.fire4(h)

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire5(h)
        h = self.fire6(h)
        h = self.fire7(h)
        h = self.fire8(h)

        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.elu(self.conv9(h))

        memory_h = chainer.Variable(h.data, volatile='AUTO')
        with chainer.no_backprop_mode():
            weight, self.memory = \
                self.apply_memory(memory_h, t, self.update_weight, self.train)

        if self.train:
            self.apply_memory.memory.data = self.memory.data

        h = F.dropout(h, ratio=0.5, train=self.train)
        h = self.conv_infer(h)
        h = F.reshape(h, (-1, self.n_class))

        h = h*weight
        self.h = h
        self.prob = F.softmax(h)

        if self.active_learn:
            t = mask_gt_for_active_learning(self.prob, t, self.xp, self.n_class)

        self.loss = F.softmax_cross_entropy(h, t)

        self.accuracy = F.accuracy(h, t)
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
