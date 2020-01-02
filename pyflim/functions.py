import numpy as np

from .misc import array_args


def fourier_coefficient(t, freq, n=1, axis=-1):
    """Compute complex Fourier coefficient from sample times.

    Parameters
    ----------
    t : array_like
        Photon arrival times.
    freq : array_like
        Frequency.
    n : array_like of ints, optional
        Number of harmonic (default is 1)
    axis : int, optional
        Axis along which the coefficient is computed.
        By default, last axis.
    """
    t, freq, n = map(np.asanyarray, (t, freq, n))
    return np.exp(1j * 2 * np.pi * freq * n * t).sum(axis=axis)


def phasor(t, freq, n=1, axis=-1):
    """Compute phasor from sample times.

    Parameters
    ----------
    t : array_like
        Photon arrival times.
    freq : array_like
        Frequency.
    n : array_like of ints, optional
        Number of harmonic (default is 1)
    axis : int, optional
        Axis along which the phasor is computed.
        By default, last axis.
    """
    t, freq, n = map(np.asanyarray, (t, freq, n))
    return np.exp(1j * 2 * np.pi * freq * n * t).mean(axis=axis)


@array_args
def phasor_reference_correction(r, r_ref):
    """Compute corrected phasor.

    Parameters
    ----------
    r, r_ref : array_like
        Phasor and reference phasor.
    """
    return r / r_ref


@array_args
def phasor_covariance(N, R1, R2):
    """Compute phasor covariance matrix.

    Parameters
    ----------
    N : array_like
        Number of counts.
    R1, R2 : array_like
        Phasor harmonics n and 2n.

    References
    ----------
    Silberberg, M., & Grecco, H. E. (2017). pawFLIM: reducing bias
    and uncertainty to enable lower photon count in FLIM experiments.
    Methods and applications in fluorescence, 5(2), 024016.
    """
    shape = np.broadcast(N, R1, R2).shape
    cov = np.empty(shape + (2, 2))
    cov[..., 0, 0] = 1 + R2.real - 2 * R1.real ** 2
    cov[..., 1, 1] = 1 - R2.real - 2 * R1.imag ** 2
    cov[..., 0, 1] = cov[..., 1, 0] = R2.imag - 2 * R1.real * R1.imag
    cov = cov / (2 * N)
    return cov


# FRET functions

@array_args
def FRET_efficiency(radius, R0):
    """Compute FRET efficiency from radius for a given Förster radius R0."""
    return 1 / (1 + (radius / R0) ** 6)


@array_args
def FRET_radius(efficiency, R0):
    """Compute FRET radius from efficiency for a given Förster radius R0."""
    return R0 * np.power(1 / efficiency - 1, 1 / 6)


@array_args
def FRET_efficiency_from_lifetime(fret_lifetime, donor_lifetime):
    """Compute FRET efficiency from FRET and donor-only states lifetimes."""
    return 1 - fret_lifetime / donor_lifetime


@array_args
def FRET_lifetime(donor_lifetime, fret_efficiency):
    """Compute FRET lifetime from donor-only lifetime and FRET efficiency."""
    return (1 - fret_efficiency) * donor_lifetime


# Single exponential functions

@array_args
def phasor_from_lifetime(tau, freq=1):
    """Compute phasor from lifetime.

    Parameters
    ----------
    tau : array_like
        Lifetime.
    freq : array_like
        Frequency.
    """
    return 1 / (1 - 1j * 2 * np.pi * freq * tau)


@array_args
def modulation_lifetime(r, freq=1):
    """Compute modulation lifetime from phasor.

    Parameters
    ----------
    r : array-like
        Phasor.
    freq : array-like
        Frequency.
    """
    return np.sqrt(1 / np.abs(r) ** 2 - 1) / (2 * np.pi * freq)


@array_args
def phase_lifetime(r, freq=1):
    """Compute phase lifetime from phasor.

    Parameters
    ----------
    r : array-like
        Phasor.
    freq : array-like
        Frequency.
    """
    return np.tan(np.angle(r)) / (2 * np.pi * freq)


@array_args
def normal_lifetime(r, freq=1):
    """Compute normal lifetime from phasor.

    Projects normally to the single-lifetime semicircle.

    Parameters
    ----------
    r : array-like
        Phasor.
    freq : array-like
        Frequency.

    References
    ----------
    Silberberg, M., & Grecco, H. E. (2017). pawFLIM: reducing bias
    and uncertainty to enable lower photon count in FLIM experiments.
    Methods and applications in fluorescence, 5(2), 024016.
    """
    return np.tan(np.angle(r - 0.5) / 2) / (2 * np.pi * freq)


# Bi-exponential functions

def semicircle_intersection(coeffs):
    """Computes the intersection of a linear polynomial with the single-lifetime semicircle.

    Parameters
    ----------
    coeffs : array_like (slope, constant)
        Coefficients of the linear polyomial.

    Returns
    -------
    phasors : array_like
        Point of intersection with semicircle ordered in
        anti-clockwise (increasing lifetime) fashion.

    Raises
    ------
    ValueError
        If there is no intersection with the semicircle.

    """
    a, b = coeffs
    disc = 1 - 4 * b * (a + b)
    if disc < 0:
        raise ValueError('Discriminant < 0, there is no intersection with the semicircle.')
    else:
        disc = np.sqrt(disc)
        x = (1 - 2 * a * b + np.array([1, -1]) * disc) / (2 * (a ** 2 + 1))
        return x + 1j * np.sqrt(x - x ** 2)


@array_args
def rotate_phasor(r, r1, r2):
    """Affine transformation mapping the biexponential segment to the real [0,1] segment.

    r, r1, r2: array-like
        Phasors, where r1 and r2 correspond to fractions 1. and 0. respectively.
    """
    return (r - r2) / (r1 - r2)


@array_args
def photon_fraction(r, r1, r2):
    """Compute a phasor's photon ratio.

    Estimated via normalized scalar projection of phasor to the biexponential line. [1]

    r, r1, r2: array-like
        Phasors, where r1 and r2 correspond to fractions 1. and 0. respectively.

    References
    ----------
    [1] Grecco, H. E., Roda-Navarro, P., & Verveer, P. J. (2009).
    Global analysis of time correlated single photon counting
    FRET-FLIM data. Optics express, 17(8), 6493-6508.
    """
    return rotate_phasor(r, r1, r2).real


@array_args
def photon_to_molecular_fraction(p, qy_ratio):
    """Convert photon to molecular fraction.

    Inverse of molecular_to_photon_fraction.

    Parameters
    ----------
    p : array_like
        Photon fraction.
    qy_ratio : array_like
        Ratio of quantum yields between states.

    Returns
    -------
    m : array_like
        Molecular fraction.
    """
    return p / (p + qy_ratio * (1 - p))


@array_args
def molecular_to_photon_fraction(m, qy_ratio):
    """Convert molecular to photon fraction.

    Inverse of photon_to_molecular_fraction.

    Parameters
    ----------
    m : array_like
        Molecular fraction.
    qy_ratio : array_like
        Ratio of quantum yields between states.

    Returns
    -------
    p : array_like
        Photon fraction.
    """
    return photon_to_molecular_fraction(m, 1 / qy_ratio)
