import sys, os
import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def show_image(img, name=None):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # name = name+'.jpg' if name is not None else 'result.jpg'
    # cv2.imwrite("/Users/naoki_shimada/Downloads/{}".format(os.path.basename(name)), img)

def devide_object(img, kenel_size=40):
    kernel = np.ones((kenel_size, kenel_size),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    return erosion


def one_ch_from_patch(img):
    patch_mean = img[125:225, 125:225].mean()
    out_img = (img-patch_mean).astype(np.uint8)
    out_img = out_img.min(axis=2)
    print('var: ',out_img.var())
    if out_img.var()>=3000:
        out_img[out_img>=out_img[125:225, 125:225].mean()*3] = 0
    elif 500<=out_img.var()<1000:
        out_img[out_img<out_img[125:225, 125:225].mean()] = 0

    return out_img.astype(np.uint8)


def binary_threshold(img):
    mean = img.mean()
    var = img.var()
    sq = np.sqrt(var)
    lower_thresh = mean - 0.05*sq
    upper_thresh = mean + 0.4*sq
    maxValue = 255
    th, drop_back = cv2.threshold(img, lower_thresh, maxValue, cv2.THRESH_BINARY)
    # th, clarify_born = cv2.threshold(gray_img, upper_thresh, maxValue, cv2.THRESH_BINARY_INV)
    # merged = np.minimum(drop_back, clarify_born)
    return drop_back


# TODO: hight universality normalization
# work wrongly with blue image
def calculate_diff_image(img, normalize=True):
    def _normalize(one_channel):
        return (one_channel - one_channel.mean())/np.sqrt(one_channel.var())

    def __change_contrast(one_channel, f=20):
        return 255.0/(1+np.exp(-f*(one_channel-128)/255))

    bgr = np.array(cv2.split(img))
    if normalize:
        bgr = np.array([_normalize(one_channel) for one_channel in bgr])
        bgr[bgr>bgr.mean(axis=0)] = 0
    idx = np.argsort(bgr.mean(axis=(1,2)))[::-1]
    # out_img = (np.abs(bgr[idx[0]]-bgr[idx[1]]) \
    #             + np.abs(bgr[idx[0]]-bgr[idx[2]])).astype(np.uint8)
    out_img = np.abs(bgr[2]).astype(np.uint8)
    out_img[out_img<np.percentile(out_img, 99)] = 0
    return out_img

def convert_one_channel_image(img, convert_type='diff', normalize=False):
    h, w, _ = img.shape
    if convert_type=='diff':
        return calculate_diff_image(img, normalize=normalize)
    elif convert_type=='gray':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif convert_type=='patch':
        return one_ch_from_patch(img)
    else:
        raise RuntimeError('undefined normalizetype type is find in transform image')


def binalize_image(img, binalize_type='otsu'):
    # # binalize(method1)
    if binalize_type=='adaptive':
        # ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C
        return cv2.adaptiveThreshold( \
                        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                        cv2.THRESH_BINARY, 11, 2)  # THRESH_BINARY_INV, THRESH_BINARY
    elif binalize_type=='otsu':
        return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    else:
        return binary_threshold(img)


def fit_ellipse(img, src_image=None, threshold_schale=100):
    h, w = img.shape
    # find corner
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    # filtered with area over (all area / 100 )
    area_th = h*w/threshold_schale
    contours_large = filter(lambda c:cv2.contourArea(c) > area_th, contours)

    result = src_image.copy() if src_image is not None  else None
    result_val = None
    for contour in contours_large:
        cnt_size = contour.size
        if cnt_size < 10: continue  # specification of openCV

        # fitting
        ellipse = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1][0], ellipse[1][1])
        # excluding axis too small or too big
        if major_axis<100 or major_axis>250: continue
        # # draw
        if src_image is not None:
            result = cv2.ellipse(result, ellipse, (0, 0, 255))
        result_val = major_axis
        break
    if src_image is not None:
        show_image(result)
    return result_val if result_val is not None else 0


def skeletonize(img):
    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    return skel


def smooth_contour(contour, n_vartex=10):
    x, y = contour.T
    # convert from numpy arrays to normal arrays
    # quiet worning(quiet=2)
    tck, u = splprep([x.flatten(), y.flatten()], s=1.0, per=1, quiet=2)
    # n_vartex = vertex(major_axis)
    u_new = np.linspace(u.min(), u.max(), n_vartex)
    x_new, y_new = splev(u_new, tck, der=0)
    # convert it back to numpy format for opencv to be able to display it
    res_array = [[[x, y]] for x, y in zip(x_new,y_new)]
    return np.asarray(res_array, dtype=np.int32)


def compute_erosioned_image(src_image, threshold_schale=100, debug=False):
    if src_image is None:
        raise RuntimeError('image data is droken.')
    h, w, _ = src_image.shape

    one_channel_image = convert_one_channel_image(
                            src_image, convert_type='diff', normalize=True)
    # bulered_image = cv2.GaussianBlur(one_channel_image, (11, 11), 0).reshape((h,w,1))
    binalized_image = binalize_image(one_channel_image, 'otsu')
    erosioned_image = devide_object(binalized_image, kenel_size=11)
    if debug:
        show_image(src_image, 'src_image')
        show_image(one_channel_image, 'one_channel_image')
        show_image(binalized_image, 'binalized_image')
        show_image(erosioned_image, 'erosioned_image')
    return erosioned_image
