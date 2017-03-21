import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ExternalMemoryAccuracy(function.Function):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    # def check_type_forward(self, in_types):
    #     type_check.expect(in_types.size() == 2)
    #     x_type, t_type = in_types

    #     type_check.expect(
    #         x_type.dtype.kind == numpy.int32,
    #         t_type.dtype == numpy.int32
    #     )

    #     t_ndim = t_type.ndim.eval()
    #     type_check.expect(
    #         x_type.ndim >= t_type.ndim,
    #         x_type.shape[0] == t_type.shape[0],
    #         x_type.shape[2: t_ndim + 1] == t_type.shape[1:]
    #     )
    #     for i in six.moves.range(t_ndim + 1, x_type.ndim.eval()):
    #         type_check.expect(x_type.shape[i] == 1)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs
        # print("dtype")
        # print(t.dtype, y.dtype)

        if self.ignore_label is not None:
            mask = (t == self.ignore_label)
            ignore_cnt = mask.sum()

            count = (y == t).sum() - ignore_cnt
            total = t.size - ignore_cnt

            if total == 0:
                return xp.asarray(0.0, dtype=xp.float32),
            else:
                return xp.asarray(float(count) / total, dtype=xp.float32),
        else:
            return xp.asarray((y == t).mean(dtype=xp.float32)),


def external_memory_accuracy(y, t, ignore_label=None):
    return ExternalMemoryAccuracy(ignore_label=ignore_label)(y, t)
