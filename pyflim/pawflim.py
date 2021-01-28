import numpy as np
from binlets import binlet

from .functions import phasor_covariance
from .misc import complex_to_real


def _phasor(N, R1, R2, **kwargs):
    r = np.zeros(np.broadcast(N, R1).shape, dtype=complex)
    r = np.divide(R1, N, where=N > 0, out=r)
    return complex_to_real(r)


def _phasor_covariance(N, R1, R2, N_thres=1):
    mask = N > N_thres
    r1 = np.divide(
        R1, N, where=mask, out=np.zeros(np.broadcast(N, R1).shape, dtype=complex)
    )
    r2 = np.divide(
        R2, N, where=mask, out=np.zeros(np.broadcast(N, R2).shape, dtype=complex)
    )
    cov = phasor_covariance(N, r1, r2, check_zero=False)
    cov[N <= N_thres] = np.diag(np.inf * np.ones(2))  # Inverse equals 0.
    return cov


_pawflim = binlet(_phasor, _phasor_covariance, False)


def pawflim(N, R1, R2, levels, p_value=0.05, N_thres=1, axes=None, mask=None):
    """pawFLIM denoising.

    Parameters
    ----------
    N : array_like
        Number of counts.
    R1, R2 : array_like
        Fourier harmonics n and 2n.
    levels : int
        Decomposition level. Must be >= 0. If == 0, does nothing.
        Sets maximum possible binning of size 2**level.
    p_value : float, optional
        Controls the level of denoising. Default is 0.05.
    N_thres : float, optional
        Pixels where N <= N_thres, automatically pass the test and are binned.
        Minimum 1.

    Returns
    -------
    N, R1 : tuple of ndarrays
        Tuple of denoised inputs.

    Other Parameters
    ----------------
    axes : tuple, optional
        Axes over which transform is applied. Default is all axes.
    mask : ndarray of bools, optional
        Marks data to denoise. Data where mask==False is not denoised.
        By default, all True.

    References
    ----------
    Silberberg, M., & Grecco, H. E. (2017). pawFLIM: reducing bias
    and uncertainty to enable lower photon count in FLIM experiments.
    Methods and applications in fluorescence, 5(2), 024016.
    """
    if N_thres < 1:
        raise ValueError
    return _pawflim(
        (
            N,
            R1,
        ),
        levels=levels,
        p_value=p_value,
        axes=axes,
        mask=mask,
        bin_args=(R2,),
        kwargs={"N_thres": N_thres},
    )
