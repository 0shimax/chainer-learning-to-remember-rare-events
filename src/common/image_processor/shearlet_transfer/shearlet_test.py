import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.data
from skimage import img_as_float
from skimage.transform import resize

from _inverse_shearlet_transform_spect import inverse_shearlet_transform_spect
from _shearlet_transform_spect import shearlet_transform_spect


def add_cbar(im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(im, cax=cax)


X = img_as_float(skimage.data.camera())
X = resize(X, (256, 256))

# compute shearlet transform
ST, Psi = shearlet_transform_spect(X)

print(X.shape)
print(ST.shape)
print(Psi.shape)

plt.imshow(X, interpolation='nearest', cmap=plt.cm.gray)
plt.colorbar()
plt.title('Frame Tightness')
plt.show()

# plt.imshow(1 - np.sum(Psi**2, -1), cmap=plt.cm.gray)
plt.imshow(ST[..., 20], interpolation='nearest', cmap=plt.cm.gray)
plt.colorbar()
plt.title('Frame Tightness')
plt.show()


XX = inverse_shearlet_transform_spect(ST, Psi)
plt.imshow(np.abs(X-XX), cmap=plt.cm.gray)
plt.colorbar()
plt.title('Transform Exactness')
plt.show()
