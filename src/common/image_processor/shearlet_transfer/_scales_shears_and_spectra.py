import numpy as np
import warnings
from meyer_shearlet import meyer_shearlet_spect, meyeraux


def _default_number_of_scales(l):
    num_of_scales = int(np.floor(0.5 * np.log2(np.max(l))))
    if num_of_scales < 1:
        raise ValueError('image to small!')
    return num_of_scales


def scales_shears_and_spectra(shape, num_of_scales=None,
                           real_coefficients=True, maxScale='max',
                           shearlet_spect=meyer_shearlet_spect,
                           shearlet_arg=meyeraux, real_real=True,
                           fftshift_spectra=True):
    if len(shape) != 2:
        raise ValueError("2D image dimensions required")

    if num_of_scales is None:
        num_of_scales = _default_number_of_scales(shape)

    # rectangular images
    if shape[1] != shape[0]:
        rectangular = True
    else:
        rectangular = False

    # for better symmetry each dimensions of the array should be odd
    shape = np.asarray(shape)
    shape_orig = shape.copy()
    shapem = np.mod(shape, 2) == 0  # True for even sized axes
    both_even = np.all(np.equal(shapem, False))
    both_odd = np.all(np.equal(shapem, True))
    shape[shapem] += 1

    if not real_coefficients:
        warnings.warn("Complex shearlet case may be buggy.  Doesn't "
                      "currently give perfect reconstruction.")

    if not (both_even or both_odd):
        # for some reason reconstruction is not exact in this case, so don't
        # allow it for now.
        raise ValueError("Mixture of odd and even array sizes is currently "
                         "unsupported.")

    # create meshgrid
    # largest value where psi_1 is equal to 1
    maxScale = maxScale.lower()
    if maxScale == 'max':
        X = 2**(2 * (num_of_scales - 1) + 1)  # = 2^(2*num_of_scales - 1)
    elif maxScale == 'min':
        X = 2**(2 * (num_of_scales - 1))  # = 2^(2*num_of_scales - 2)
    else:
        raise ValueError('Wrong option for maxScale, must be "max" or "min"')

    xi_x_init = np.linspace(0, X, (shape[1] + 1) / 2)
    xi_x_init = np.concatenate((-xi_x_init[-1:0:-1], xi_x_init), axis=0)
    if rectangular:
        xi_y_init = np.linspace(0, X, (shape[0] + 1) / 2)
        xi_y_init = np.concatenate((-xi_y_init[-1:0:-1], xi_y_init), axis=0)
    else:
        xi_y_init = xi_x_init

    # create grid, from left to right, bottom to top
    [xi_x, xi_y] = np.meshgrid(xi_x_init, xi_y_init[::-1], indexing='xy')

    # cones
    C_hor = np.abs(xi_x) >= np.abs(xi_y)  # with diag
    C_ver = np.abs(xi_x) < np.abs(xi_y)

    # number of shears: |-2^j,...,0,...,2^j| = 2 * 2^j + 1
    # now: inner shears for both cones:
    # |-(2^j-1),...,0,...,2^j-1|
    # = 2 * (2^j - 1) + 1
    # = 2^(j+1) - 2 + 1 = 2^(j+1) - 1
    # outer scales: 2 ("one" for each cone)
    # shears for each scale: hor: 2^(j+1) - 1, ver: 2^(j+1) - 1, diag: 2
    #  -> hor + ver + diag = 2*(2^(j+1) - 1) +2 = 2^(j + 2)
    #  + 1 for low-pass
    shears_per_scale = 2**(np.arange(num_of_scales) + 2)
    num_of_all_shears = 1 + shears_per_scale.sum()

    # init
    Psi = np.zeros(tuple(shape) + (num_of_all_shears, ))
    # frequency domain:
    # k  2^j 0 -2^j
    #
    #     4  3  2  -2^j
    #      \ | /
    #   (5)- x -1  0
    #      / | \
    #              2^j
    #
    #        [0:-1:-2^j][-2^j:1:2^j][2^j:-1:1] (not 0)
    #           hor          ver        hor
    #
    # start with shear -2^j (insert in index 2^j+1 (with transposed
    # added)) then continue with increasing scale. Save to index 2^j+1 +- k,
    # if + k save transposed. If shear 0 is reached save -k starting from
    # the end (thus modulo). For + k just continue.
    #
    # then in time domain:
    #
    #  2  1  8
    #   \ | /
    #  3- x -7
    #   / | \
    #  4  5  6
    #

    # lowpass
    Psi[:, :, 0] = shearlet_spect(xi_x, xi_y, np.NaN, np.NaN, real_coefficients,
                                 shearlet_arg, scaling_only=True)

    # loop for each scale
    for j in range(num_of_scales):
        # starting index
        idx = 2**j
        start_index = 1 + shears_per_scale[:j].sum()
        shift = 1
        for k in range(-2**j, 2**j + 1):
            # shearlet spectrum
            P_hor = shearlet_spect(xi_x, xi_y, 2**(-2 * j), k * 2**(-j),
                                  real_coefficients, shearlet_arg)
            if rectangular:
                P_ver = shearlet_spect(xi_y, xi_x, 2**(-2 * j), k * 2**(-j),
                                      real_coefficients, shearlet_arg)
            else:
                # the matrix is supposed to be mirrored at the counter
                # diagonal
                # P_ver = fliplr(flipud(P_hor'))
                P_ver = np.rot90(P_hor, 2).T  # TODO: np.conj here too?
            if not real_coefficients:
                # workaround to cover left-upper part
                P_ver = np.rot90(P_ver, 2)

            if k == -2**j:
                Psi[:, :, start_index + idx] = P_hor * C_hor + P_ver * C_ver
            elif k == 2**j:
                Psi_idx = start_index + idx + shift
                Psi[:, :, Psi_idx] = P_hor * C_hor + P_ver * C_ver
            else:
                new_pos = np.mod(idx + 1 - shift, shears_per_scale[j]) - 1
                if(new_pos == -1):
                    new_pos = shears_per_scale[j] - 1
                Psi[:, :, start_index + new_pos] = P_hor
                Psi[:, :, start_index + idx + shift] = P_ver

                # update shift
                shift += 1

    # generate output with size shape_orig
    Psi = Psi[:shape_orig[0], :shape_orig[1], :]

    # modify spectra at finest scales to obtain really real shearlets
    # the modification has only to be done for dimensions with even length
    if real_coefficients and real_real and (shapem[0] or shapem[1]):
        idx_finest_scale = (1 + np.sum(shears_per_scale[:-1]))
        scale_idx = idx_finest_scale + np.concatenate(
            (np.arange(1, (idx_finest_scale + 1) / 2 + 1),
             np.arange((idx_finest_scale + 1) / 2 + 2, shears_per_scale[-1])),
            axis=0)
        scale_idx = scale_idx.astype(np.int)
        if shapem[0]:  # even number of rows -> modify first row:
            idx = slice(1, shape_orig[1])
            Psi[0, idx, scale_idx] = 1 / np.sqrt(2) * (
                Psi[0, idx, scale_idx] +
                Psi[0, shape_orig[1] - 1:0:-1, scale_idx])
        if shapem[1]:  # even number of columns -> modify first column:
            idx = slice(1, shape_orig[0])
            Psi[idx, 0, scale_idx] = 1 / np.sqrt(2) * (
                Psi[idx, 0, scale_idx] +
                Psi[shape_orig[0] - 1:0:-1, 0, scale_idx])

    if fftshift_spectra:
        # Note: changed to ifftshift so roundtrip tests pass for odd sized
        # arrays
        Psi = np.fft.ifftshift(Psi, axes=(0, 1))
    return Psi
