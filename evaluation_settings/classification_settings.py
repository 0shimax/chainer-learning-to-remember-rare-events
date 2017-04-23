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
in_ch = 1  # calculate_in_ch(converse_gray, detect_edge)
generate_comment = False
crop = False
resize = True
gpu = -1 if platform.system()==base_params.local_os_name else 1

# 'squeeze_net_dilate', 'squeeze_net_external_memory',
# 'squeeze_net_with_feature_vals', 'squeeze_net_external_long_term_memory'
use_net = 'squeeze_net_external_long_term_memory'
n_class = 10  # number of class is 2, if you use ne_class classifier.
crop_size = 352
normalize_type = 'LCN'  # 'ZCA', 'LCN', 'SLT'
im_norm_type = set_normalizer(normalize_type)
model_module = set_net(use_net)
experiment_criteria = ''  # '_59k_SGD_ndicay_YUVlcnYUV_f8_nloss_shuf_flip_rot_scaling_shift'
output_path = os.path.join(base_params.data_root_path+'/results', use_net+experiment_criteria)
initial_model = os.path.join( \
    base_params.data_root_path+'/results'+'/'+use_net+experiment_criteria, \
    'model_iter_xxx')  # cp_model_iter_584566
resume = os.path.join( \
    base_params.data_root_path+'/results'+'/'+use_net+experiment_criteria, \
    'snapshot_iter_xxx')
aug_flags = {'do_scale':False, 'do_flip':False,
             'change_britghtness':False, 'change_contrast':False,
             'do_shift':False, 'do_rotate':False,
             'do_blur':False}

# aug_flags = {'do_scale':False, 'do_flip':False,
#              'change_britghtness':True, 'change_contrast':False,
#              'do_shift':False, 'do_rotate':False,
#              'do_blur':False}

# reset training params
base_params.training_params.lr = 1e-3  # SqueezeNet: 1e-3, other: 1e-4
base_params.training_params.decay_epoch = 300
# base_params.training_params.optimizer = 'NesterovAG'  # 'AdaDelta', NesterovAG
base_params.training_params.optimizer = 'Adam'  # 'AdaDelta', NesterovAG
# base_params.training_params.updater_type = 'parallel'
# base_params.training_params.batch_size = 5
base_params.training_params.clip_grad = True
# base_params.training_params.iter_type = 'multi'


# a body of args
train_args = \
    {
        'train': True,
        'with_feature_val': True if use_net=='squeeze_net_with_feature_vals' else False,
        'active_learn': False,
        'generate_comment': generate_comment,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': n_class,
        'in_ch': in_ch,
        # 'image_dir_path': base_params.data_root_path+'/phase1_stage1',
        # 'image_pointer_path': base_params.data_root_path+'/phase1_image_pointer_extracted_with_trained_model',
        # 'labels_file_path': base_params.data_root_path+'/phase1_stage1_labels_extracted_with_trained_model',

        # 'image_dir_path': base_params.data_root_path+'/6k_rivised',
        # 'image_pointer_path': base_params.data_root_path+'/6k_revised_image_pointer19',
        # 'labels_file_path': base_params.data_root_path+'/6k_revised_labels19',

        # 'image_dir_path': base_params.data_root_path+'/6k_rivised',
        # 'image_pointer_path': base_params.data_root_path+'/6k_revised_image_pointer',
        # 'labels_file_path': base_params.data_root_path+'/6k_revised_labels',

        # 'image_dir_path': base_params.data_root_path,
        # 'image_pointer_path': base_params.data_root_path+'/image_pointer_for_cross_val_train1',
        # 'labels_file_path': base_params.data_root_path+'/labels_for_cross_val_train1',

        'image_dir_path': base_params.data_root_path+'/sys_val400',
        'image_pointer_path': base_params.data_root_path+'/val400_image_paths',
        'labels_file_path': base_params.data_root_path+'/val400_labels',

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
        'with_feature_val': True if use_net=='squeeze_net_with_feature_vals' else False,
        'active_learn': False,  # never use in test.
        'generate_comment': generate_comment,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': n_class,
        'in_ch': in_ch,
        # 'image_dir_path': base_params.data_root_path+'/phase1_stage1',
        # 'image_pointer_path': base_params.data_root_path+'/phase1_image_pointer',
        # 'labels_file_path': base_params.data_root_path+'/phase1_stage1_labels',

        'image_dir_path': base_params.data_root_path+'/sys_val400',
        'image_pointer_path': base_params.data_root_path+'/val400_image_paths',
        'labels_file_path': base_params.data_root_path+'/val400_labels',

        # 'image_dir_path': base_params.data_root_path+'/sys_val28',
        # 'image_pointer_path': base_params.data_root_path+'/val28_image_paths',
        # 'labels_file_path': base_params.data_root_path+'/val28_labels',

        # 'image_dir_path': base_params.data_root_path+'/6k_rivised',
        # 'image_pointer_path': base_params.data_root_path+'/6k_revised_image_pointer19',
        # 'labels_file_path': base_params.data_root_path+'/6k_revised_labels19',

        # 'image_dir_path': base_params.data_root_path+'/20170117_revised_13000_Images',
        # 'image_pointer_path': base_params.data_root_path+'/20170117_revised_13000_image_pointer',
        # 'labels_file_path': base_params.data_root_path+'/20170117_revised_13000_labels',

        # 'image_dir_path': base_params.data_root_path,
        # 'image_pointer_path': base_params.data_root_path+'/image_pointer_for_cross_val_test1',
        # 'labels_file_path': base_params.data_root_path+'/labels_for_cross_val_test1',

        'weights_file_path': base_params.data_root_path+'/Learning_Data_Blind_weights',  # never use in test.
        'output_path': output_path,
        'initial_model': initial_model,
        'resume': '',  # never use in test.
        'im_norm_type': im_norm_type,
        'archtecture': model_module,
        'converse_gray': converse_gray,
        'detect_edge': detect_edge,
        'do_resize': resize,
        'crop_params': {'flag':crop, 'size': crop_size},
        'multiple': base_params.mult_dir[use_net],  # total stride multiple
        'aug_params': {'do_augment':False,
                        'params': dict(base_params.augmentation_params, **aug_flags),
                    },
        'importance_sampling': False,
        'shuffle': False,
        'training_params': base_params.training_params,
        'token_args': base_params.token_args,
    }


def get_args(args_type='train'):
    if args_type=='train':
        check_types(train_args)
        return edict(train_args)
    else:
        check_types(test_args)
        return edict(test_args)
