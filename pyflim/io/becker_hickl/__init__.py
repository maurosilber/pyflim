"""Adapted from JediFLIM by Klaus Schuermann and tdflim by Peter Verveer."""

import numpy as np

from ...flimds import UncorrectedFLIMds
from ..functions import fourier_image, histogram
from .bh_header import read_header_spc
from .bh_numba import interpret_AI, read_records


class SPC(UncorrectedFLIMds):
    """Becker&Hickl .spc file"""

    def load_TTTR(self):
        channel, dtime, truetime = read_records(
            self.filename,
            self.nb_records,
            self.recstart,
            self.syncrate,
            self.resolution,
        )

        self.x, self.y, self.f, self.dtime, _, self.pixY = interpret_AI(
            channel,
            dtime,
            truetime,
            0,
            0,
            self.lsm_frame,
            self.lsm_line_start,
            self.lsm_pixel_start,
            self.pixel_dwell_time,
        )
        self.pixX = self.x.max() + 1

    def __init__(self, fname, tac_range):
        self.filename = fname
        self.header, self.recstart = read_header_spc(fname)

        self.syncrate = 1 / self.header["macro_clock"]

        # NOTE: Not known from .spc header file, need to get them from some meta-data.
        # For now set it in your code by hand after initializing. This is just a guess.
        # self.tac_range = 25e-9
        self.tac_range = tac_range

        # The number of TAC channels is always 12bit, i.e. 4096
        self.num_tac_channels = 4096

        # Will be set later from TAC range and channels
        self.resolution = self.tac_range / self.num_tac_channels

        self.nb_records = self.header["num_records"]

        # We will try to guess them unless you reset them in your code using meta-data
        # self.pixX = 0
        # self.pixY = 0
        self.scannertype = "AI"

        # These are the default settings in Imspector. It could also be, that no pixel clocks are
        # recorded (Imspector can deal with it). In this case you need to set the pixel dwell time
        # below.
        self.lsm_frame = 0x04
        self.lsm_line_start = 0x02
        self.lsm_pixel_start = 0x01

        # Not needed. If it is set, it can be used in case pixX is set to calculate the pixel dwell
        # time
        self.lsm_line_stop = -1

        # Change from 0 if you want it used to interpret data from BH hardware without pixel marker
        self.pixel_dwell_time = 0

        # From tdFLIM class
        self.TAC_period = 1 / (self.resolution * self.syncrate)
        self.binning = 1
        self.tacbinmax = -1
        self.type = "bh"

        self.load_TTTR()

    @property
    def num_TAC_bins(self):
        return self.num_tac_channels

    @property
    def frequency(self):
        return self.syncrate

    def histogram(self, mask=None):
        hist = histogram(self.dtime, self.x, self.y, self.num_TAC_bins, mask=mask)
        bins = np.arange(hist.size) * self.resolution
        return bins, hist

    def fourier_image(self, harmonics, mask=None):
        return fourier_image(
            (self.pixY, self.pixX),
            harmonics,
            self.dtime,
            self.x,
            self.y,
            self.num_TAC_bins,
            self.TAC_period,
            mask=mask,
        )
