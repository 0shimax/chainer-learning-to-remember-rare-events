import sys, os
sys.path.append('./src/common')
sys.path.append('./src/common/image_processor')
sys.path.append('./src/net')
sys.path.append('./src/net/RAM')
sys.path.append('./src/net/caption_generator')
sys.path.append('./src/net/feedback')
sys.path.append('./experiment_settings')
from mini_batch_loader import DatasetPreProcessor
from important_serial_iterator import ImportantSerialIterator
from copy_model import copy_model

import chainer
import chainer.functions as F
from chainer import serializers
from chainer import cuda, Variable
from chainer import Reporter, report, report_scope
from chainer import optimizers, serializers, training
from chainer.training import extensions
import importlib


class EasyTrainer(object):
    def __init__(self, args, settings_type):
        self.args = args
        self.settings_type = settings_type

    @staticmethod
    def prepare_model(args):
        model = getattr(
            importlib.import_module(args.archtecture.module_name), \
                                    args.archtecture.class_name) \
                                    (args.n_class, args.in_ch)
        if os.path.exists(args.initial_model):
            print('Load model from', args.initial_model, file=sys.stderr)
            serializers.load_npz(args.initial_model, model)

        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            model.to_gpu()
        model.train = True
        model.n_class = args.n_class
        m_eval = model.copy()
        m_eval.train = False

        model.active_learn = args.active_learn
        return model, m_eval

    def select_optimizer(self):
        if self.args.training_params.optimizer=='RMSpropGraves':
            return chainer.optimizers.RMSpropGraves(self.args.training_params.lr)
        elif self.args.training_params.optimizer=='Adam':
            return chainer.optimizers.Adam()
        elif self.args.training_params.optimizer=='AdaDelta':
            return chainer.optimizers.AdaDelta()
        elif self.args.training_params.optimizer=='NesterovAG':
            return chainer.optimizers.NesterovAG(self.args.training_params.lr)
        elif self.args.training_params.optimizer=='MomentumSGD':
            return chainer.optimizers.MomentumSGD(self.args.training_params.lr)

    def prepare_optimizer(self, model):
        optimizer = self.select_optimizer()
        optimizer.setup(model)
        # print("optimizer.target.params()")
        # print(optimizer.target.__dict__)
        # optimizer.target.__dict__ = {k: v for k, v in optimizer.target.__dict__.items() if "memory_" not in k}
        # print(optimizer.target.__dict__)
        # optimizer.target._params = (param for param in optimizer.target.params() if "memory_" not in param.name)
        # print([param for param in optimizer.target.params()])
        if self.args.training_params.weight_decay:
            optimizer.add_hook(chainer.optimizer.WeightDecay( \
                self.args.training_params.weight_decay))
        if self.args.training_params.lasso:
            optimizer.add_hook(chainer.optimizer.Lasso( \
                self.args.training_params.weight_decay))
        if self.args.training_params.clip_grad:
            optimizer.add_hook(chainer.optimizer.GradientClipping( \
                self.args.training_params.clip_value))
        return optimizer

    def prepare_dataset(self):
        mode_settings = importlib.import_module(self.settings_type)
        train_args = mode_settings.get_args('train')
        # load dataset
        train_mini_batch_loader = DatasetPreProcessor(train_args)
        test_mini_batch_loader = DatasetPreProcessor(mode_settings.get_args('test'))
        print("---set mini_batch----------")

        if train_args.importance_sampling:
            print("importance----------")
            train_it = ImportantSerialIterator( \
                                    train_mini_batch_loader, \
                                    train_args.training_params.batch_size, \
                                    shuffle=train_args.shuffle, \
                                    p=np.loadtxt(train_args.weights_file_path))
        else:
            if train_args.training_params.iter_type=='multi':
                iterator = chainer.iterators.MultiprocessIterator
            else:
                iterator = chainer.iterators.SerialIterator
            train_it = iterator( \
                            train_mini_batch_loader, \
                            train_args.training_params.batch_size, \
                            shuffle=train_args.shuffle)

        val_batch_size = 1
        val_it = iterator( \
                    test_mini_batch_loader, \
                    val_batch_size, repeat=False, shuffle=False)
        return train_it, val_it, train_mini_batch_loader.__len__()

    def prepare_updater(self, train_it, optimizer):
        if self.args.training_params.updater_type=='standerd':
            return training.StandardUpdater( \
                train_it, optimizer, device=self.args.gpu)
        elif self.args.training_params.updater_type=='parallel':
            return training.ParallelUpdater( \
                train_it, optimizer, devices={'main': 1, 'second': 0})

    def run_trainer(self):
        # load model
        model, model_for_eval = self.prepare_model(self.args)
        print("---set model----------")

        # Setup optimizer
        optimizer = self.prepare_optimizer(model)
        print("---set optimzer----------")

        # load data
        train_it, val_it, train_data_length = self.prepare_dataset()
        print("---set data----------")

        updater = self.prepare_updater(train_it, optimizer)
        print("---set updater----------")

        evaluator_interval = self.args.training_params.report_epoch, 'epoch'
        snapshot_interval = self.args.training_params.snapshot_epoch, 'epoch'
        log_interval = self.args.training_params.report_epoch, 'epoch'

        trainer = training.Trainer( updater, \
            (self.args.training_params.epoch, 'epoch'), out=self.args.output_path)
        trainer.extend( \
            extensions.Evaluator(val_it, model_for_eval, device=self.args.gpu), \
            trigger=evaluator_interval)
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
        trainer.extend(extensions.snapshot_object( \
            model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
        if self.args.training_params.optimizer!='Adam' \
                    and self.args.training_params.optimizer!='AdaDelta':
            trainer.extend(extensions.ExponentialShift( \
                'lr', self.args.training_params.decay_factor), \
                trigger=(self.args.training_params.decay_epoch, 'epoch'))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.PrintReport([ \
            'epoch', 'iteration', 'main/loss', 'validation/main/loss', \
            'main/accuracy', 'validation/main/accuracy', \
            ]), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=1))
        print("---set trainer----------")

        if os.path.exists(self.args.resume):
            print('resume trainer:{}'.format(self.args.resume))
            # Resume from a snapshot
            serializers.load_npz(self.args.resume, trainer)

        trainer.run()
