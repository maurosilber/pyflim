import numba as nb
import numpy as np


def histogram(dtime, x, y, num_TAC_bins, mask=None):
    """Compute the histogram of the measurement.

    Parameters
    ----------
    dtime, x, y: array_like
        TAC bin, x and y coordinates of photons
    num_TAC_bins : int
        Number of bins with non-zero photons (dtime.max + 1)
    mask : ndarray of bools of shape (y_dim, x_dim), optional
        Image will be calculated only where mask == True.
    """
    hist = np.zeros(num_TAC_bins, dtype=int)
    _histogram(hist, dtime, x, y, mask=mask)
    return hist


def fourier_image(image_shape, harmonics, dtime, x, y, num_TAC_bins, TAC_period, mask=None):
    """Compute complex fourier coefficients for every pixel in the image.

    Parameters
    ----------
    image_shape : tuple of ints (y_dim, x_dim)
        Output image shape.
    harmonics : array_like
        Harmonics to compute.
    dtime, x, y: array_like
        TAC bin, x and y coordinates of photons
    num_TAC_bins : int
        Number of bins with non-zero photons (dtime.max + 1)
    TAC_period : float
        Number of bins corresponding to time between laser pulses.
        In theory should be the same as bins, but due to jitter in
        the laser stability some photons can arrive later.
    mask : ndarray of bools of shape (y_dim, x_dim), optional
        Image will be calculated only where mask == True.

    Returns
    -------
    image : ndarray of shape (num_harmonics, y_dim, x_dim)
    """
    image = np.zeros((len(harmonics), *image_shape), dtype=complex)
    _fourier_image(image, complex_exp(harmonics, num_TAC_bins, TAC_period), dtime, x, y, mask=mask)
    return image


def complex_exp(harmonics, num_TAC_bins, TAC_period):
    """Generate a matrix for calculation of Fourier coefficients.

    Parameters
    ----------
    harmonics : array_like
        Harmonics to compute.
    num_TAC_bins : int
        Number of bins with non-zero photons (dtime.max + 1)
    TAC_period : float
        Number of bins corresponding to time between laser pulses.
        In theory should be the same as bins, but due to jitter in
        the laser stability some photons can arrive later.

    Returns
    -------
    """
    harmonics = np.asarray(harmonics)
    phi = np.arange(num_TAC_bins) * 2 * np.pi / TAC_period
    return np.exp(1j * harmonics * phi[:, None])  # FLIM sign convention.


@nb.njit
def _fourier_image(image, complex_exp, dtime, x, y, mask=None):
    """Numba-compiled function to compute fourier_image.

    Parameters
    ----------
    image : complex ndarray of shape (num_harmonics, y_dim, x_dim)
        Output image.
    complex_exp : ndarray
        Complex wave of dimensions (num_TAC_bins, num_harmonics)
    """

    num_harm = complex_exp.shape[1]
    if mask is None:
        for dt, xi, yi in zip(dtime, x, y):
            for h in range(num_harm):
                image[h, yi, xi] += complex_exp[dt, h]
    else:
        for dt, xi, yi in zip(dtime, x, y):
            if mask[yi, xi]:
                for h in range(num_harm):
                    image[h, yi, xi] += complex_exp[dt, h]


@nb.njit
def _histogram(hist, dtime, x, y, mask=None):
    """Numba-compiled function to compute histogram.

    Parameters
    ----------
    hist : ndarray of num_TAC_bins length
        Output histogram.
    """
    if mask is None:
        for dt in dtime:
            hist[dt] += 1
    else:
        for dt, xi, yi in zip(dtime, x, y):
            if mask[yi, xi]:
                hist[dt] += 1
