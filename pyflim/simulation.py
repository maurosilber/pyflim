import numpy as np


def _exponential_rv(t, tau, T):
    """Generate truncated exponential random variable from uniform [0, 1) random variable.

    Parameters
    ----------
    t : array-like
        Uniform [0, 1) random variable.
    tau : array-like
        Lifetime.
    T : array-like
        Truncation parameter.
    """
    return -tau * np.log(1 - t * (1 - np.exp(-T / tau)))


## VER PARAMETRO N
def single_lifetime(tau, N, T):
    """Generate single exponential samples.

    Parameters
    ----------
    tau : array-like
        Lifetime.
    N : array-like
        Number of samples.
    T : array-like
        Size of measurement window (inverse of measurement frequency).

    Returns
    -------

    """
    t = np.random.random(size=N)
    t = _exponential_rv(t, tau, T)
    return t


def dual_lifetime(tau1, tau2, p, N, T):
    """Generate bi-exponential samples.

    Parameters
    ----------
    tau1, tau2 : array-like
        Lifetime.
    p : array-like
        Photon fraction.
    N : array-like
        Number of samples.
    T : array-like
        Size of measurement window (inverse of measurement frequency).

    Returns
    -------

    """
    return multi_lifetime((tau1, tau2), (p, 1 - p), N, T)


def multi_lifetime(taus, p, N, T):
    """Generate multi-exponential samples.

    Parameters
    ----------
    taus : list or array-like
        List of lifetimes.
    p : list or array-like
        List of photon fractions.
    N : array-like
        Number of samples.
    T : array-like
        Size of measurement window (inverse of measurement frequency).

    Returns
    -------

    """
    p = np.asarray(p)
    p = p / p.sum(axis=0)  # Normalize photon fractions

    b = np.random.multinomial(1, p, N)
    b = np.argmax(b, axis=-1)

    t = np.random.random(size=N)
    for k, tau in enumerate(taus):
        cond = b == k
        t[cond] = _exponential_rv(t[cond], tau, T)

    return t
