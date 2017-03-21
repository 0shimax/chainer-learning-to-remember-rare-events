import numpy
import six

import chainer
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer.functions import where
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class ModLSTM(chainer.links.LSTM):
    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.
        Args:
            x (~chainer.Variable): A new batch from the input sequence.
        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.
        """
        if self.upward.has_uninitialized_params:
            in_size = x.size // x.shape[0]
            self.upward._initialize_params(in_size)
            self._initialize_params()

        batch = x.shape[0]
        lstm_in = self.upward(x)
        h_rest = None
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == 0:
                h_rest = self.h
            elif h_size < batch:
                msg = ('The batch size of x must be equal to or less than the '
                       'size of the previous state h.')
                raise TypeError(msg)
            elif h_size > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
                lstm_in += self.lateral(h_update)
            else:
                lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((batch, self.state_size), dtype=x.dtype),
                volatile='auto')
        # self.c, y = lstm.lstm(self.c, lstm_in)

        c, y = lstm.lstm(self.c, lstm_in)
        enable = (x.data != -1)
        self.c = where(enable, c, self.c)
        if self.h is not None:
            y = where(enable, y, self.h)

        if h_rest is None:
            self.h = y
        elif len(y.data) == 0:
            self.h = h_rest
        else:
            self.h = concat.concat([y, h_rest], axis=0)

        return y
