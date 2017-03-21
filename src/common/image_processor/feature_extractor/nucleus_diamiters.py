import sys, os
sys.path.append('./src/common/image_processor/feature_extractor')
import cv2
import numpy as np
import pandas as pd
from operator import itemgetter
from collections import defaultdict
from feature_extractor_utils import (show_image,
                                    smooth_contour,
                                    compute_erosioned_image)


def judge_max_min(max_val, min_val, candidate_val):
    if candidate_val is not None:
        max_val = candidate_val if candidate_val>max_val else max_val
        min_val = candidate_val if candidate_val<min_val else min_val
    return min_val, max_val


def judge_min(min_val, candidate_val):
    if candidate_val is not None:
        min_val = candidate_val if candidate_val<min_val else min_val
    return min_val


def vertex(major_axis):
    if major_axis<100:
        return 5
    else:
        return 10


def calculate_ellipse_axis(contour):
    ellipse = cv2.fitEllipse(contour)
    major_axis = max(ellipse[1][0], ellipse[1][1])
    minor_axis = min(ellipse[1][0], ellipse[1][1])
    return minor_axis, major_axis


def compute_max_min_diamiter(scanning_axis, measure_axis, scan_gap=10):
    left_top = (scanning_axis.min(), measure_axis.min())
    right_bottom = (scanning_axis.max(), measure_axis.max())
    mergin = (right_bottom[0] - left_top[0])//10
    pre_point = None
    max_val, min_val = 0, float('inf')
    for x in range(left_top[0]+mergin, right_bottom[0]-mergin+1):
        idx = np.where((scanning_axis>=x-scan_gap)&(scanning_axis<=x+scan_gap))[0]
        if len(idx)%2!=0: continue
        for i_idx in range(0, len(idx),2):
            start, end = idx[i_idx], idx[i_idx+1]
            val = np.abs(measure_axis[start]-measure_axis[end])
            if val<10: continue
            min_val, max_val = judge_max_min(max_val, min_val, val)
    return min_val, max_val,


def calculate_one_cnt_diameter(min_v, max_v, contour, major_axis_sub, \
                                low_threshold, high_threshold, src_image=None):
    cnt_size = contour.size
    if cnt_size < 10:  # specification of openCV
        return min_v, max_v

    try:
        contour = smooth_contour(contour)
        # fitting
        minor_axis, major_axis = calculate_ellipse_axis(contour)
    except:
        # fitting
        minor_axis, major_axis = calculate_ellipse_axis(contour)
        return int(minor_axis), int(major_axis)

    # excluding axis too small or too big
    if major_axis<low_threshold-major_axis_sub or \
                        major_axis>high_threshold:
        return min_v, max_v

    contour = contour.reshape((contour.shape[0], 2))
    if src_image is not None:
        cv2.drawContours(src_image,[contour],-1,(0,255,0),3)
        show_image(src_image)
    x_len = max(contour[:,0]) - min(contour[:,0])
    y_len = max(contour[:,1]) - min(contour[:,1])
    if y_len >= x_len:
        contour_y = np.array(sorted(contour, key=itemgetter(1,0)))
        # t_min_v, t_max_v = compute_max_min_diamiter(contour_y, scan_axis='x')
        t_min_v, t_max_v = compute_max_min_diamiter(contour_y[:,1], contour_y[:,0])
    else:
        contour_x = np.array(sorted(contour, key=itemgetter(0,1)))
        # t_min_v, t_max_v = compute_max_min_diamiter(contour_x, scan_axis='y')
        t_min_v, t_max_v = compute_max_min_diamiter(contour_x[:,0], contour_x[:,1])
    min_v, max_v = judge_max_min(max_v, min_v, t_min_v)
    min_v, max_v = judge_max_min(max_v, min_v, t_max_v)
    return min_v, max_v


def compute_diameter(img, src_image=None, threshold_schale=100,
                                    low_threshold=95, high_threshold=300):
    h, w = img.shape
    # find corner
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    # filtered with area over (all area / 100 )
    area_th = h*w/threshold_schale
    contours_large = list(filter(lambda c:cv2.contourArea(c) > area_th, contours))

    result = src_image.copy() if src_image is not None  else None
    min_v, max_v = float('inf'), 0
    major_axis_sub = 0
    while max_v==0 and major_axis_sub<low_threshold-1:
        for contour in contours_large:
            min_v, max_v = calculate_one_cnt_diameter(min_v, max_v, contour, \
                                            major_axis_sub, low_threshold, \
                                            high_threshold, src_image=src_image)
        major_axis_sub += 10
    max_diamiter = max(max_v, min_v)
    if max_diamiter==float('inf'):
        max_diamiter = 0
    min_diamiter = min(max_v, min_v)
    return min_diamiter, max_diamiter,


def compute_nucleus_diameter(src_image, threshold_schale=100, debug=False):
    image = compute_erosioned_image(src_image, threshold_schale, debug)
    if debug:
        return compute_diameter(image, src_image, threshold_schale=threshold_schale)
    else:
        return compute_diameter(image, threshold_schale=threshold_schale)

if __name__=='__main__':
    import subprocess

    image_dir = './data/6k_rivised'
    # image_dir = './data/20170117_revised_13000_Images'
    # image_dir = './data/sys_val400'
    # image_dir = './data/sys_val28'

    cmd = 'find {} -name "*.jpg"'.format(image_dir)
    process = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,)
    b_out, _ = process.communicate()
    out = b_out.decode('utf-8').rstrip().split('\n')

    dict_results = defaultdict(list)
    for image_path in out:
        # print(image_path)
        label = image_path.split('/')[-1].split('_')[0]
        image = cv2.imread(image_path)
        min_diam, max_diam = \
            compute_nucleus_diameter(image, threshold_schale=100, debug=False)
        dict_results['label'].append(label)
        dict_results['min_diam'].append(min_diam)
        dict_results['max_diam'].append(max_diam)
        # print(label, min_diam, max_diam)

    data = pd.DataFrame(dict_results)
    print(data.groupby('label').agg([np.mean, np.median]))
