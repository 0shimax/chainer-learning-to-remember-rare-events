import sys
sys.path.append('./src/common/image_processor/shearlet_transfer')
import cv2
import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from sklearn.utils.extmath import fast_dot
from sklearn.utils import check_array
from math import sqrt

from _inverse_shearlet_transform_spect import inverse_shearlet_transform_spect
from _shearlet_transform_spect import shearlet_transform_spect


class ImageNormalizer(object):
    def __init__(self):
        pass

    def zca_whitening(self, image, eps):
        """
        N = 1
        X = image[:,:].reshape((N, -1)).astype(np.float64)

        X = check_array(X, dtype=[np.float64], ensure_2d=True, copy=True)

        # Center data
        self.mean_ = np.mean(X, axis=0)
        print(X.shape)
        X -= self.mean_

        U, S, V = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        zca_matrix = U.dot(np.diag(1.0/np.sqrt(np.diag(S) + 1))).dot(U.T) #ZCA Whitening matrix

        return fast_dot(zca_matrix, X).reshape(image.shape)   #Data whitening
        """
        image = self.local_contrast_normalization(image)
        N = 1
        X = image.reshape((N, -1))

        pca = PCA(whiten=True, svd_solver='full', n_components=X.shape[-1])
        transformed = pca.fit_transform(X)  # return U
        pca.whiten = False
        zca = fast_dot(transformed, pca.components_+eps) + pca.mean_
        # zca = pca.inverse_transform(transformed)
        return zca.reshape(image.shape)


    def local_contrast_normalization(self, image, color='RGB'):
        # TODO: refactoring
        h, w, ch = image.shape
        if ch==3 and color=='YUV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        image = image.astype(np.uint8)
        if color=='YUVandRGB':
            cn_channels = tuple(cv2.equalizeHist(d_ch) if idx==0 else d_ch \
                                for idx, d_ch in enumerate(cv2.split(image)))
        else:
            cn_channels = tuple(cv2.equalizeHist(d_ch) for d_ch in cv2.split(image))

        if len(cn_channels)==3:
            image = cv2.merge(cn_channels)
            if color=='YUVandRGB':
                image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
                cn_channels = tuple(cv2.equalizeHist(d_ch) for d_ch in cv2.split(image))
                return cv2.merge(cn_channels)
            else:
                return image
        elif len(cn_channels)==1:
            return cn_channels[0].reshape((h, w, 1))

    def global_contrast_normalization(self, image, args=None):
        mean = np.mean(image)
        var = np.var(image)
        return (image-mean)/float(sqrt(var))

    def shearlet_transform(self, image, args=None):
        def __restoration(im2d, st, psi):
            # cv2.imshow('raw image', st[...,40])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # xx = inverse_shearlet_transform_spect(st, psi)
            # return np.abs(im2d - xx)
            return st[...,30:].sum(axis=2)

        image = image.astype(np.uint8)
        im2d_st_psi_sets = tuple((d_ch,)+shearlet_transform_spect(d_ch) \
                                            for d_ch in cv2.split(image))

        restored_im_per_chanel = tuple(__restoration(im2d, ST, Psi) \
                                        for im2d, ST, Psi in im2d_st_psi_sets)
        if len(restored_im_per_chanel)==1:
            return restored_im_per_chanel[0].reshape(image.shape[:2]+(1,))
        return cv2.merge(restored_im_per_chanel)
