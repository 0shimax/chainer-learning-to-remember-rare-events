import os, sys
sys.path.append('./experiment_settings')
from easydict import EasyDict as edict
import platform
from base_settings import (get_base_params,
                            set_normalizer,
                            set_net)
from type_checker import check_val_type, check_types


# common params
base_params = get_base_params()


# hand making params
gpu = -1 if platform.system()==base_params.local_os_name else 1
use_net = 'squeeze_net_dilate'
n_class = 19  # number of class is 2, if you use ne_class classifier.
normalize_type = 'LCN'  # 'ZCA', 'LCN'
im_norm_type = set_normalizer(normalize_type)
model_module = set_net(use_net)
experiment_criteria = '_59k_SGD_ndicay_YUVlcnYUV_f8_nloss_shuf_flip_rot_scaling_shift'  # '_lcn_f8_nloss_shuf_flip_rot_scaling_shift'  # '_20161003'
output_path = os.path.join(base_params.data_root_path+'/results', use_net+experiment_criteria)
initial_model = os.path.join( \
    base_params.data_root_path+'/results'+'/'+use_net+experiment_criteria, \
    'cp_model_iter_584566')  # cp_model_iter_381000
resume = os.path.join( \
    base_params.data_root_path+'/results'+'/'+use_net+experiment_criteria, \
    'snapshot_iter_xxx')


visualize_args = \
    {
        # check 00007_0_0.jpg.
        # size=72, stride=11 is best.
        'size': 104,  # size>=32, for pyramid spacial pooling. Default 112.
        'stride': 11,
        'in_ch': 3,  # channel of input images.
        'percentile': 99,  # percentile value
        'patch_batchsize': 4096,
        'view_type': 'infer',  # select 'infer' or 'gt'.

        'image_dir_path': base_params.data_root_path+'/sys_val400',
        'image_pointer_path': base_params.data_root_path+'/val400_image_paths',
        'labels_file_path': base_params.data_root_path+'/val400_labels',
    }


# a body of args
base_args = \
    {
        'train': False,
        'active_learn': False,
        'with_feature_val': True if use_net=='squeeze_net_with_feature_vals' else False,
        'generate_comment': False,
        'debug_mode': False,
        'converse_gray': False,
        'gpu': gpu,
        'n_class': n_class,
        'output_path': output_path,
        'initial_model': initial_model,
        'im_norm_type': im_norm_type,
        'archtecture': model_module,
        'detect_edge': False,
        'do_resize': True,
        'crop_params': {'flag':False, 'size': 352},
        'multiple': base_params.mult_dir[use_net],  # total stride multiple
        'aug_params': {'do_augment':False},
        'token_args': base_params.token_args,
    }


def get_args():
    args = edict(dict(base_args, **visualize_args))
    check_types(args)
    return args
