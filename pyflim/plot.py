import numpy as np
from matplotlib import colors, patches
from matplotlib import pyplot as plt
from matplotlib import transforms

from . import functions


def semicircle(ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    return ax.add_patch(patches.Arc((0.5, 0), 1, 1, theta2=180.0, **kwargs))


def phasor_scatter(r, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    return ax.scatter(r.real, r.imag, **kwargs)


def phasor_plot(
    R, bins=100, hist_range=((0, 1), (0, 0.6)), ax=None, log=True, **kwargs
):
    if ax is None:
        ax = plt.gca()
    H, *edges = np.histogram2d(
        R.real.ravel(), R.imag.ravel(), bins=bins, range=hist_range
    )
    X = np.meshgrid(*edges)
    H = np.ma.masked_less_equal(H, 0)
    norm = (
        colors.LogNorm(vmin=1, vmax=H.max())
        if log
        else colors.Normalize(vmin=0, vmax=H.max())
    )
    return ax.pcolormesh(*X, H.T, norm=norm, **kwargs)


def confidence_ellipse(N, r1, r2, ax=None, n_std=3.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if ax is None:
        ax = plt.gca()

    cov = functions.phasor_covariance(N, r1, r2)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patches.Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = r1.real, r1.imag

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
