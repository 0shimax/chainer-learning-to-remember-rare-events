import numpy
from chainer.dataset import iterator


class ImportantSerialIterator(iterator.Iterator):
    def __init__(self, dataset, batch_size, repeat=True, shuffle=True, p=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        if shuffle:
            self._order = numpy.random.choice(range(len(dataset)), len(dataset), replace=repeat, p=p)
            # self._order = numpy.random.permutation(len(dataset))
        else:
            self._order = None

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        if self._order is None:
            batch = self.dataset[i:i_end]
        else:
            batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    numpy.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch += list(self.dataset[:rest])
                    else:
                        batch += [self.dataset[index]
                                  for index in self._order[:rest]]
                self.current_position = rest
            else:
                self.current_position = N

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)
