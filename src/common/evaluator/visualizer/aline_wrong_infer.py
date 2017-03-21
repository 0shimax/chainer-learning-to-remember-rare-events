import sys
sys.path.append('./src/common')
import os
import random
import PIL.Image
import PIL.ImageStat
import matplotlib.pyplot as plt
from base_info import get_base_info


clsval2labstr = get_base_info(fetch_type='clsval2sysmlabstr')


def get_attended_file_name( \
        raw_file_path, gt_label, infer_label):
    raw_file_name = os.path.basename(raw_file_path)
    name, extention = raw_file_name.split('.')
    attended_file_name = name+'_'+str(gt_label)+'_'+str(infer_label)+'.'+extention
    return attended_file_name


def judge_wrong_type(gt_prob, gt_label, infer_label):
    if gt_label!=infer_label:
        return 'wrong inference'
    else:
        return 'low confidence'


def set_plt(img, n_column, label, prob, n, data_type='gt'):
    plt.subplot( 1, n_column, n+1)
    plt.axis('off')
    plt.imshow(img)
    base_title = 'raw' if data_type=='gt' else 'attention'
    label = int(label)
    plt.title('{0:s}({1:s}={2:s}, prob={3:.5f})'.format( \
                    base_title, data_type, clsval2labstr[label], float(prob)))
    if n==n_column-1:
        plt.show()
    return (n+1)%n_column


def get_label_prob_string(label, prob):
    prob = str(float(prob))
    return '{0:s}={1:.5f}'.format(clsval2labstr[int(label)], float(prob))


def show_pair(one_result, raw_image_dir_path, attended_image_dir_path, \
                                                    n_column=2, figsize=(18,5)):
    n = 0
    wrong_file_name = one_result[0]

    result_labels = one_result[1:5]
    gt_label = result_labels[0]
    infer_label = result_labels[1]

    result_prob = one_result[-4:]
    gt_prob = result_prob[0]
    infer_prob = result_prob[1]

    # wrong_file_name, gt_prob, infer_prob, gt_label, infer_label = one_result

    attended_file_name = \
        get_attended_file_name(wrong_file_name, gt_label, infer_label)
    if n==0:
        plt.figure(figsize=figsize)
    wrong_type = judge_wrong_type(gt_prob, gt_label, infer_label)
    print(wrong_file_name+'({})'.format(wrong_type))

    s_top3_pair = [get_label_prob_string(label, prob) \
                            for label, prob in zip(result_labels[1:], result_prob[1:])]
    print('Inferred Probabilities TOP3: {}'.format(', '.join(s_top3_pair)))

    raw_img = PIL.Image.open(raw_image_dir_path+'/'+wrong_file_name)
    attended_img = PIL.Image.open(attended_image_dir_path+'/'+attended_file_name)

    n = set_plt(raw_img, n_column, gt_label, gt_prob, n, data_type='gt')
    n = set_plt(attended_img, n_column, infer_label, infer_prob, n, data_type='infer')
    # n = set_plt(attended_img, n_column, infer_label, gt_prob, n, data_type='infer')


def show_all_pairs(wrong_and_low_conf_images_params, \
                            raw_image_dir_path, attended_image_dir_path):
    for one_result in wrong_and_low_conf_images_params:
        show_pair(one_result, raw_image_dir_path, attended_image_dir_path)


if __name__=='__main__':
    USER_BASE_PATH = '/Users/naoki_shimada'
    PJ_BASE_PATH = os.path.join(USER_BASE_PATH, 'projects/da-sysmex')
    RESULT_STORE_PATH = os.path.join(PJ_BASE_PATH, 'data/results/squeeze_net_dilate/low_confidence_image_path')
    RAW_DATA_DIR_PATH = os.path.join(PJ_BASE_PATH, 'data/sys_val400')
    ATTENDED_IMAGE_DIR_PATH = os.path.join(USER_BASE_PATH, 'Downloads/visualized_attention_images')

    result_data = []
    with open(RESULT_STORE_PATH, 'r') as result_infile:
        for line in result_infile:
            result_data.append( line.rstrip().split(',') )  # wrong_file_name, prob, gt, infer

    show_all_pairs(result_data, RAW_DATA_DIR_PATH, ATTENDED_IMAGE_DIR_PATH)
