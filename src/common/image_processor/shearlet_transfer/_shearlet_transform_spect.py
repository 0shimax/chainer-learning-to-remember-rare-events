import numpy as np
import cv2

from meyer_shearlet import meyer_shearlet_spect, meyeraux

from _scales_shears_and_spectra import scales_shears_and_spectra
from _fft import fftshift, ifftshift, fftn, ifftn


def shearlet_transform_spect(image_2d, Psi=None, num_of_scales=None,
                           real_coefficients=True, maxScale='max',
                           shearlet_spect=meyer_shearlet_spect,
                           shearlet_arg=meyeraux, real_real=True):

    # parse input
    if (image_2d.ndim != 2) or np.any(np.asarray(image_2d.shape) <= 1):
        raise ValueError("2D image required")

    # compute spectra
    if Psi is None:
        l = image_2d.shape
        if num_of_scales is None:
            num_of_scales = int(np.floor(0.5 * np.log2(np.max(l))))
            if num_of_scales < 1:
                raise ValueError('image to small!')
        Psi = scales_shears_and_spectra(l, num_of_scales=num_of_scales,
                                     real_coefficients=real_coefficients,
                                     shearlet_spect=meyer_shearlet_spect,
                                     shearlet_arg=meyeraux)

    # shearlet transform
    uST = Psi * fftn(image_2d)[..., np.newaxis]
    ST = ifftn(uST, axes=(0, 1))

    # due to round-off errors the imaginary part is not zero but very small
    # -> neglect it
    if real_coefficients and real_real and np.isrealobj(image_2d):
        ST = ST.real

    return (ST, Psi)
