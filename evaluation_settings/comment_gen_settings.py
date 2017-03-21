import os, sys
sys.path.append('./experiment_settings')
from easydict import EasyDict as edict
import platform
from base_settings import (get_base_params,
                            calculate_in_ch,
                            set_normalizer,
                            set_net)
from type_checker import check_val_type, check_types


# common params
base_params = get_base_params()


# hand making params
debug_mode = False
converse_gray = False
detect_edge = False
in_ch = calculate_in_ch(converse_gray, detect_edge)
generate_comment = True
crop = True
resize = False
gpu = -1 if platform.system()==base_params.local_os_name else 1
use_net = 'review_net'
n_class = 19  # number of class is 2, if you use ne_class classifier.
crop_size = 352
normalize_type = 'LCN'  # 'ZCA', 'LCN'
im_norm_type = set_normalizer(normalize_type)
model_module = set_net(use_net)
experiment_criteria = ''  # '_20161003'
output_path = os.path.join(base_params.data_root_path+'/results', use_net+experiment_criteria)
initial_model = os.path.join( \
    base_params.data_root_path+'/results'+'/'+use_net+experiment_criteria, \
    'model_iter_xxx')
resume = os.path.join( \
    base_params.data_root_path+'/results'+'/'+use_net+experiment_criteria, \
    'snapshot_iter_xxx')
aug_flags = {'do_scale':False, 'do_flip':True,
             'change_britghtness':False, 'change_contrast':False,
             'do_shift':False, 'do_rotate':False}


# reset training params
base_params.training_params.lr = 1e-3
base_params.training_params.snapshot_epoch = 30
base_params.training_params.clip_grad = True
base_params.training_params.clip_value = 5.
# base_params.training_params.iter_type = 'multi'


# a body of args
train_args = \
    {
        'train': True,
        'active_learn': False,
        'generate_comment': generate_comment,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': n_class,
        'in_ch': in_ch,
        'image_dir_path': base_params.data_root_path+'/Learning_Data_Blind',
        'image_pointer_path': base_params.data_root_path+'/trainin_image_paths5',
        'labels_file_path': base_params.data_root_path+'/Learning_Data_Blind_labels5',
        'weights_file_path': base_params.data_root_path+'/Learning_Data_Blind_weights',
        'output_path': output_path,
        'initial_model': initial_model,
        'resume': resume,
        'im_norm_type': im_norm_type,
        'archtecture': model_module,
        'converse_gray': converse_gray,
        'detect_edge': detect_edge,
        'do_resize': resize,
        'crop_params': {'flag':crop, 'size': crop_size},
        'multiple': base_params.mult_dir[use_net],  # total stride multiple
        'aug_params': {'do_augment':True,
                       'params': dict(base_params.augmentation_params, **aug_flags),
                      },
        'importance_sampling': False,
        'shuffle': True,  # data shuffle in SerialIterator
        'training_params': base_params.training_params,
        'token_args': base_params.token_args,
    }

test_args = \
    {
        'train': False,
        'generate_comment': generate_comment,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': n_class,
        'in_ch': in_ch,
        'image_dir_path': base_params.data_root_path+'/Learning_Data_Blind',
        'image_pointer_path': base_params.data_root_path+'/trainin_image_paths5',
        'labels_file_path': base_params.data_root_path+'/Learning_Data_Blind_labels5',

        # 'image_dir_path': base_params.data_root_path+'/sys_val400',
        # 'image_pointer_path': base_params.data_root_path+'/val400_image_paths',
        # 'labels_file_path': base_params.data_root_path+'/val400_labels',

        # 'image_dir_path': base_params.data_root_path+'/sys_val28',
        # 'image_pointer_path': base_params.data_root_path+'/val28_image_paths',
        # 'labels_file_path': base_params.data_root_path+'/val28_labels',
        'output_path': output_path,
        'initial_model': initial_model,
        'im_norm_type': im_norm_type,
        'archtecture': model_module,
        'converse_gray': converse_gray,
        'detect_edge': detect_edge,
        'do_resize': resize,
        'crop_params': {'flag':crop, 'size': crop_size},
        'multiple': base_params.mult_dir[use_net],  # total stride multiple
        'aug_params': {'do_augment':False},
        'token_args': base_params.token_args,
    }


def get_args(args_type='train'):
    if args_type=='train':
        check_types(train_args)
        return edict(train_args)
    else:
        check_types(test_args)
        return edict(test_args)
