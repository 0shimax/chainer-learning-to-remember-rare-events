import numpy
from chainer.dataset import iterator
import subprocess
import sys
sys.path.append('./src/common')
from base_info import get_base_info


class SerialIteratorForMatching(iterator.Iterator):
    def __init__(self, dataset, batch_size, file_pointer, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat

        # NC_, OTH_ nessesary to joint
        self.klass_list = list(get_base_info('labstr2clsval').keys())
        each_pointers, max_len = self.extract_class_file(file_pointer)

        if shuffle:
            self._order = numpy.random.permutation(len(dataset))
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

    def extract_class_file(self, file_pointer):
        result = []
        other_idices = []
        max_len = 0
        for idx, klass in enumerate(self.klass_list):
            cmd = 'cat {}|grep /{}_'.format(file_pointer, klass)
            process = subprocess.Popen(cmd, shell=True,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                    )
            b_out, err = process.communicate()
            out = b_out.decode('utf-8').rstrip()
            out = out.split('\n')
            result.append(out)
            if klass=='NC' or klass=='OTH':
                other_idices.append(idx)
            if len(out) > max_len:
                max_len = len(out)

        other_res = []
        for oi in other_idices:
            other_res += result[oi]
        max_len = len(other_res) if len(other_res) > max_len else max_len
        [result.pop(i) for i in other_idices]
        result.insert(other_idices[0], other_res)
        return result, max_len

    def pic_idx_each_class(self, each_pointers, max_len, repeat):
        result = []
        total_pointer_len = 0
        for idx, pointer in enumerate(each_pointers):
            idx = numpy.random.choice(range(max_len), max_len, replace=repeat)
            idx += total_pointer_len

            total_pointer_len += len(pointer)
        result += idx.tolist()
