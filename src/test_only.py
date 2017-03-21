import sys
sys.path.append('./src/common')
sys.path.append('./src/common/image_processor')
sys.path.append('./src/net')
sys.path.append('./experiment_settings')
from mini_batch_loader import DatasetPreProcessor
from classification_settings import get_args
from trainer_utils import EasyTrainer

import chainer
from chainer import serializers, Variable

import sys, os, math, inspect
import numpy as np
from collections import defaultdict


def test(model, args):
    sum_accuracy = 0
    sum_loss     = 0
    omit_counter = 0
    heat_map = defaultdict(dict)

    val_it, test_data_size, pairs = prepare_dataset(args)
    print("------test data size")
    print(test_data_size)
    out_file = os.path.join(args.output_path, args.low_confidence_image_output_file_name)
    with open(out_file, 'w') as low_conf_of:
        for idx, batch in enumerate(val_it):
            raw_x, raw_t = np.array([batch[0][0]], dtype=np.float32), \
                                np.array([batch[0][1]],dtype=np.int32)
            gt_label = int(batch[0][1])
            omit_counter += 1 if gt_label==-1 else 0

            if args.gpu>-1:
                chainer.cuda.get_device(args.gpu).use()
                x = chainer.Variable(chainer.cuda.to_gpu(raw_x), volatile=True)
                t = chainer.Variable(chainer.cuda.to_gpu(raw_t), volatile=True)
            else:
                x = chainer.Variable(raw_x, volatile=True)
                t = chainer.Variable(raw_t, volatile=True)

            model(x, t)
            sum_loss += model.loss.data
            sum_accuracy += model.accuracy.data

            # image restoration
            if args.gpu>-1:
                infer_label = chainer.cuda.to_cpu(model.h.data.argmax())
                prob = chainer.cuda.to_cpu(model.prob.data[0])
                top3_idx = np.argsort(-prob)[:3]
            else:
                infer_label = model.h.data.argmax()
                top3_idx = np.argsort(-model.prob.data[0])[:3]
            infer_label = int(infer_label)
            heat_map[gt_label][infer_label] = heat_map[gt_label].get(infer_label,0)+1

            gt_prob = model.prob.data[0, gt_label]

            if (gt_label==infer_label and gt_prob<args.low_conf_thresh) \
                                                    or gt_label!=infer_label:
                file_name = os.path.basename(pairs[idx][0])
                out_list = [file_name, gt_label] + top3_idx.tolist()

                out_list += [gt_prob]
                out_list += [model.prob.data[0, infer_label] for \
                                                    infer_label in top3_idx]
                s = ','.join(map(str, out_list))
                low_conf_of.write(s+'\n')

    out_file = os.path.join(args.output_path, args.output_file_name)
    with open(out_file, 'w') as of:
        s = ','+','.join(map(str,args.label_exist))
        of.write(s+'\n')

        for gt_label in args.label_exist:
            out_list = []
            out_list.append(gt_label)
            for infer_label in args.label_exist:
                cnt = heat_map[gt_label].get(infer_label,0)
                out_list.append(cnt)
            s = ','.join(map(str, out_list))
            of.write(s+'\n')

    print("test mean accuracy {}".format(sum_accuracy/(test_data_size-omit_counter)))


def prepare_dataset(args):
    # load dataset
    test_mini_batch_loader = DatasetPreProcessor(args)
    val_it = chainer.iterators.SerialIterator( \
                            test_mini_batch_loader, \
                            1, repeat=False, shuffle=False)
    return val_it, test_mini_batch_loader.__len__(), test_mini_batch_loader.pairs


def main(args):
    _, model_eval = EasyTrainer.prepare_model(get_args('train'))
    test(model_eval, args)


if __name__ == '__main__':
    args = get_args('test')
    args.label_exist = list(range(19))
    args.output_file_name = 'result_heat_map.csv'
    args.low_confidence_image_output_file_name = 'low_confidence_image_path'
    args.low_conf_thresh = 0.75
    main(args)
