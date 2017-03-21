import numpy as np
import cv2

from meyer_shearlet import meyer_shearlet_spect, meyeraux
from _scales_shears_and_spectra import scales_shears_and_spectra
from _fft import fftshift, ifftshift, fftn, ifftn


def inverse_shearlet_transform_spect(ST, Psi=None, maxScale='max',
                                  shearlet_spect=meyer_shearlet_spect,
                                  shearlet_arg=meyeraux):
    if Psi is None:
        # num_of_scales
        # possible: 1, 4, 8, 16, 32,
        # -> -1 for lowpass
        # -> divide by for (1, 2, 4, 8,
        # -> +1 results in a 2^# number -> log returns #
        num_of_scales = int(np.log2((ST.shape[-1] - 1)/4 + 1))

        # real_coefficients
        real_coefficients = True

        # real_real
        real_real = True

        # compute spectra
        Psi = scales_shears_and_spectra((ST.shape[0], ST.shape[1]),
                                     num_of_scales=num_of_scales,
                                     real_coefficients=real_coefficients,
                                     real_real=real_real,
                                     shearlet_spect=meyershearlet_spect,
                                     shearlet_arg=meyeraux)

    # inverse shearlet transform
    image_2d = fftn(ST, axes=(0, 1)) * Psi
    image_2d = image_2d.sum(axis=-1)
    image_2d = ifftn(image_2d)

    if np.isrealobj(ST):
        image_2d = image_2d.real

    return image_2d
