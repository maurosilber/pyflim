import numpy as np

from .functions import (
    modulation_lifetime,
    normal_lifetime,
    phase_lifetime,
    phasor_from_lifetime,
    photon_fraction,
    photon_to_molecular_fraction,
)


class UncorrectedFLIMds:
    """Abstract FLIM dataset."""

    @property
    def frequency(self):
        raise NotImplementedError

    def histogram(self, mask=None):
        """Compute histogram.

        Parameters
        ----------
        mask : array_like
            Histogram is only computed where mask == True.

        Returns
        -------
        bins, hist : tuple of ndarrays
        """
        raise NotImplementedError

    def fourier_image(self, harmonics, mask=None):
        """Computes complex Fourier coefficients for specified harmonics.

        Parameters
        ----------
        harmonics : array_like
            Harmonics to compute.
        mask : array_like
            Coefficients are only computed where mask == True.
        """
        raise NotImplementedError

    def phasor_image(self, harmonics, mask=None, ret_N=False):
        """Computes the phasor image for specified harmonics.

        Parameters
        ----------
        harmonics : array_like
            Harmonics to compute. If harmonics[0] is not 0,
            it is included to compute the number of counts.
        mask : array_like
            Coefficients are only computed where mask == True.
        ret_N : bool
            If True, return number of counts.

        Note: keywords are passed to self.fourier_image

        Returns
        -------
        phasor_image : array_like of shape (harmonics, *image.shape)

        If ret_N, returns (counts, phasor_image).
        """
        if harmonics[0] != 0:
            harmonics = (0, *harmonics)

        R = self.fourier_image(harmonics, mask=mask)
        N, R = R[0].real, R[1:]
        r = np.zeros_like(R)
        r = np.divide(R, N, where=N > 0, out=r)

        if ret_N:
            return N, r
        else:
            return r

    def mean_phasor(self, harmonics, mask=None):
        """Computes the weighted mean phasor."""
        N, R = self.phasor_image(harmonics, mask=mask, ret_N=True)
        return np.sum(N * R) / np.sum(N)

    def to_corrected(self, irf, bg):
        """Calculate and return a corrected FLIM dataset.

        Parameters
        ----------
        irf, bg : UncorrectedFLIMds
            Datasets for IRF and background.

        Returns
        -------
        CorrectedFLIMds
        """
        return CorrectedFLIMds(self, irf, bg)


class Constant(UncorrectedFLIMds):
    def __init__(self, coeffs, harmonics=None):
        if harmonics is None:
            harmonics = range(len(coeffs))
        self.coeffs = dict(zip(harmonics, coeffs))

    def fourier_image(self, harmonics, mask=None):
        return np.fromiter(
            (self.coeffs[h] for h in harmonics), dtype=complex, count=len(harmonics)
        )


class CorrectedFLIMds:
    """An IRF and background-corrected FLIM dataset."""

    def __init__(self, ufds, irf, bg):
        self.ufds = ufds
        self.irf = irf
        self.bg = bg

    def phasor_image(self, harmonics, mask=None):
        """Calculate normalized fourier coefficient."""
        if harmonics[0] != 0:
            harmonics = (0, *harmonics)

        R = self.ufds.fourier_image(harmonics, mask=mask)
        N, R = R[0].real, R[1:]
        Rb = self.bg.fourier_image(harmonics, mask=mask)
        Nb, Rb = Rb[0].real, Rb[1:]
        irf = self.irf.phasor_image(harmonics, mask=mask)
        return (R - Rb) / (N - Nb) / irf

    def mlt_image(self, harmonics, mask=None):
        """Calculate modulation lifetime image."""
        return modulation_lifetime(
            self.phasor_image(harmonics, mask=mask), self.ufds.frequency
        )

    def plt_image(self, harmonics, mask=None):
        """Calculate phase lifetime image."""
        return phase_lifetime(
            self.phasor_image(harmonics, mask=mask), self.ufds.frequency
        )

    def nlt_image(self, harmonics, mask=None):
        """Calculate normal lifetime image."""
        return normal_lifetime(
            self.phasor_image(harmonics, mask=mask), self.ufds.frequency
        )


class FRETFLIMds(CorrectedFLIMds):
    def __init__(self, ufds, irf, bg, r_fret, r_donor):
        super().__init__(ufds, irf, bg)
        self.r_fret = r_fret
        self.r_donor = r_donor

    @classmethod
    def from_lifetimes(cls, ufds, irf, bg, lt_fret, lt_donor):
        r_fret, r_donor = phasor_from_lifetime((lt_fret, lt_donor), freq=ufds.frequency)
        return cls(ufds, irf, bg, r_fret, r_donor)

    @classmethod
    def from_cfds(cls, cfds, r_fret, r_donor):
        """Loads the dataset from a corrected flim dataset."""
        return cls(cfds.ufds, cfds.irf, cfds.bg, r_fret, r_donor)

    def p_image(self, harmonics, mask=None):
        """Calculate and return a photon fraction image."""
        return photon_fraction(
            self.phasor_image(harmonics, mask=mask), self.r_fret, self.r_donor
        )

    def m_image(self, harmonics, mask=None):
        """Calculate and return a molecular fraction image."""
        fret_lifetime, donor_lifetime = normal_lifetime(
            (self.r_fret, self.r_donor), self.ufds.frequency
        )
        return photon_to_molecular_fraction(
            self.p_image(harmonics, mask=mask), fret_lifetime / donor_lifetime
        )
