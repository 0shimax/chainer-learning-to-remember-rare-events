import sys, os
sys.path.append('./src/common/image_processor/feature_extractor')
import cv2
import numpy as np
from feature_extractor_utils import fit_ellipse, compute_erosioned_image


# Pseudo, obtain mager diameter of nucleus elliptical fitting.
def compute_cell_diameter(src_image, threshold_schale=100, debug=False):
    image = compute_erosioned_image(src_image, threshold_schale, debug)
    if debug:
        return fit_ellipse(image, src_image)
    else:
        return fit_ellipse(image)

if __name__=='__main__':
    import subprocess

    image_dir = './data/20170117_revised_13000_Images'
    # image_dir = './data/sys_val400'
    # image_dir = './data/sys_val28'

    cmd = 'find {} -name "*.jpg"'.format(image_dir)
    process = subprocess.Popen(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,)
    b_out, _ = process.communicate()
    out = b_out.decode('utf-8').rstrip().split('\n')

    for image_path in out:
        image = cv2.imread(image_path)
        print(compute_cell_diameter(image, threshold_schale=100, debug=True))
