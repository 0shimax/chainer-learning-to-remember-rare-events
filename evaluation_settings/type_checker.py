bool_val_list = ['train', 'with_feature_val', 'active_learn', \
                'generate_comment', 'debug_mode', 'converse_gray'
                'detect_edge', 'do_resize', 'importance_sampling', 'shuffle']
numeric_val_list = ['gpu', 'in_ch', 'in_ch', 'size', 'stride', 'percentile', \
                    'patch_batchsize', ]
string_val_list = ['image_dir_path', 'image_pointer_path', 'labels_file_path', \
                    'weights_file_path', 'output_path', 'initial_model', \
                    'resume', 'view_type']
dict_val_list = ['im_norm_type', 'archtecture', 'crop_params', 'aug_params', \
                'training_params', 'token_args']

check_vals_dir = {
        'bool': bool_val_list,
        'numeric': numeric_val_list,
        'str': string_val_list,
        'dict': dict_val_list,
    }


def check_types(args):
    for checking_type, target_keys in check_vals_dir.items():
        for key in target_keys:
            if key in args:
                check_val_type(checking_type, args[key], key)


def check_val_type(checking_type, val, val_name):
    eval('check_{}'.format(checking_type))(val, val_name)


def get_var_name(val):
    for k, v in globals().items():
        if id(v) == id(val):
            return k

def check_bool(param, val_name):
    if not isinstance(param, bool):
        raise TypeError("{} is must be boolean.".format(val_name))


def check_numeric(param, val_name):
    if not isinstance(param, int) or isinstance(param, bool):
        raise TypeError("{} is must be numeric.".format(val_name))


def check_str(param, val_name):
    if not isinstance(param, str):
        raise TypeError("{} is must be string.".format(val_name))


def check_dict(param, val_name):
    if not isinstance(param, dict):
        raise TypeError("{} is must be dict".format(val_name))
