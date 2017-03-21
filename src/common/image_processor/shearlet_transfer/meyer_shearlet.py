import numpy as np


def meyeraux(x):
    # Auxiliary def values.
    y = np.polyval([-20, 70, -84, 35, 0, 0, 0, 0], x) * (x >= 0) * (x <= 1)
    y += (x > 1)
    return y


def meyer_bump(x, meyeraux_func=meyeraux):
    int1 = meyeraux_func(x) * (x >= 0) * (x <= 1)
    y = int1 + (x > 1)
    return y


def bump(x, meyeraux_func=meyeraux):
    y = meyer_bump(1+x, meyeraux_func)*(x <= 0) + \
        meyer_bump(1-x, meyeraux_func)*(x > 0)
    y = np.sqrt(y)
    return y


def meyer_scaling(x, meyeraux_func=meyeraux):
    xa = np.abs(x)

    # Compute support of Fourier transform of phi.
    int1 = ((xa < 1/2))
    int2 = ((xa >= 1/2) & (xa < 1))

    # Compute Fourier transform of phi.
    # phihat = int1 * np.ones_like(xa)
    # phihat = phihat + int2 * np.cos(np.pi/2*meyeraux_func(2*xa-1))
    phihat = int1 + int2 * np.cos(np.pi/2*meyeraux_func(2*xa-1))

    return phihat


def _meyer_helper(x, real_coefficients=True, meyeraux_func=meyeraux):
    if real_coefficients:
        xa = np.abs(x)
    else:
        # consider left and upper part of the image due to first row and column
        xa = -x

    int1 = ((xa >= 1) & (xa < 2))
    int2 = ((xa >= 2) & (xa < 4))

    psihat = int1 * np.sin(np.pi/2*meyeraux_func(xa-1))
    psihat = psihat + int2 * np.cos(np.pi/2*meyeraux_func(1/2*xa-1))

    y = psihat
    return y


def meyer_wavelet(x, real_coefficients=True, meyeraux_func=meyeraux):
    y = np.sqrt(np.abs(_meyer_helper(x, real_coefficients, meyeraux_func))**2 +
                np.abs(_meyer_helper(2*x, real_coefficients, meyeraux_func))**2)
    return y


def meyer_shearlet_spect(x, y, a, s, real_coefficients=True,
                       meyeraux_func=meyeraux, scaling_only=False):
    if scaling_only:
        # cones
        C_hor = np.abs(x) >= np.abs(y)  # with diag
        C_ver = np.abs(x) < np.abs(y)
        Psi = (meyer_scaling(x, meyeraux_func) * C_hor +
               meyer_scaling(y, meyeraux_func) * C_ver)
        return Psi

    # compute scaling and shearing
    y = s * np.sqrt(a) * x + np.sqrt(a) * y
    x = a * x

    # set values with x=0 to 1 (for division)
    xx = (np.abs(x) == 0) + (np.abs(x) > 0)*x

    # compute spectrum
    Psi = meyer_wavelet(x, real_coefficients, meyeraux_func) * \
        bump(y/xx, meyeraux_func)
    return Psi


def meyer_smooth_shearlet_spect(x, y, a, s, real_coefficients=True,
                             meyeraux_func=meyeraux, scaling_only=False):
    if scaling_only:
        Psi = meyer_scaling(x, meyeraux_func) * meyer_scaling(y, meyeraux_func)
        return Psi

    if not real_coefficients:
        raise ValueError('Complex shearlets not supported for smooth Meyer '
                         'shearlets!')

    # compute scaling and shearing
    asy = s * np.sqrt(a) * x + np.sqrt(a) * y
    y = a * y
    x = a * x

    # set values with x=0 to 1 (for division)
    # xx = (np.abs(x)==0) + (np.abs(x)>0)*x

    # compute spectrum
    W = np.sqrt((meyer_scaling(2**(-2)*x, meyeraux_func) *
                 meyer_scaling(2**(-2)*y, meyeraux_func))**2 -
                (meyer_scaling(x, meyeraux_func) *
                 meyer_scaling(y, meyeraux_func))**2)
    Psi = W * bump(asy/x, meyeraux_func)

    # reset NaN to 0
    Psi[np.isnan(Psi)] = 0
    return Psi
