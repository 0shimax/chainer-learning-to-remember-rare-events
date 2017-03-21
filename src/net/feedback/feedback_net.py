import sys
sys.path.append('./src/net/feedback')
import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L

import numpy as np
import cv2
from conv_lstm import ConvLSTM


class RecurrentBlock(chainer.ChainList):
    def __init__(self, in_ch, out_ch, kernel, stride, pad, n_layers):
        super().__init__()
        self.add_link(ConvLSTM(in_ch, out_ch, kernel, stride=stride, pad=pad))
        for i in range(n_layers):
            self.add_link(ConvLSTM(out_ch, out_ch, kernel, stride=1, pad=pad))
        self.n_layers = n_layers

    def __call__(self, x, h=None):
        h_flag = h is None

        h_list = []
        # TODO: depth=1の入力には何も工夫を入れなくて良いのか?
        h_curr = self[0](x)  # if h_flag else self[0](x+h[0])
        h_list.append(h_curr)
        for i in range(1,self.n_layers):
            if h is not None:
            h_curr = self[i](h_curr) if h_flag else self[i](h_curr+h[i])
            h_list.append(h_curr)
        return h_list

    def clear(self):
        for i in range(self.n_layers):
            self[i].reset_state()


class IterateBlock(chainer.Chain):
    def __init__(self, in_ch, out_ch, kernel, stride, pad, n_layers,
                                                        iter_time, n_class):
        super().__init__(
            rec = RecurrentBlock(in_ch, out_ch, kernel, stride, pad, n_layers),
            conv_infer = L.Convolution2D(out_ch*21, n_class, 1, pad=0),
        )
        self.iter_time = iter_time
        self.n_class = n_class

    def __call__(self, x, t):
        h_lists = []
        self.loss = 0

        for i in range(self.iter_time):
            h_list_p2 = h_lists[i-2] if i>0 and i%2==0 else None
            h_list = self.rec(x, h=h_list_p2)
            h_lists.append(h_list)

            h = F.spatial_pyramid_pooling_2d(h_list[-1], 3, F.MaxPooling2D)
            h = F.elu(h)
            h = self.conv_infer(h)
            h = F.reshape(h, (-1, self.n_class))
            self.loss += F.softmax_cross_entropy(h, t)

            # x = F.elu(h_list[-1])
        self.h = h
        return h_list[-1]

    def clear(self):
        self.rec.clear()


class FeedbackNet(chainer.Chain):
    def __init__(self, n_class, in_ch, n_e=128, n_h=256, g_size=13, n_step=8, scale=3, var=7.5):
        super().__init__(
            conv1 = L.Convolution2D(in_ch, 16, 3, 1, 1),  # preprocess
            rec1 = IterateBlock(16, 16, 3, 1, 1, 3, 4, n_class),
            rec2 = IterateBlock(16, 32, 3, 2, 1, 3, 4, n_class),
            rec3 = IterateBlock(32, 64, 3, 2, 1, 3, 4, n_class),
            rec4 = IterateBlock(64, 64, 3, 1, 1, 3, 4, n_class),
        )
        self.train = True
        self.n_class = n_class

    def clear(self):
        self.loss = 0.
        self.accuracy = None
        # iter IterateBlock number
        for i_rec in range(1, 5):
            r = getattr(self, 'rec{}'.format(i_rec))
            r.clear()

    def __call__(self, x, t):
        self.clear()
        x.volatile = not self.train

        h = self.conv1(x)
        h = self.rec1(h, t)
        h = self.rec2(h, t)
        h = self.rec3(h, t)
        self.rec4(h, t)
        h = self.rec4.h

        self.h = h
        self.prob = F.softmax(h)
        self.loss =   self.rec1.loss \
                    + self.rec2.loss \
                    + self.rec3.loss \
                    + self.rec4.loss

        self.accuracy = F.accuracy(h, t)
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
