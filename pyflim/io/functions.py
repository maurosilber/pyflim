from enum import Enum

import numba as nb
import numpy as np

from . import pq_header, pq_numba as pq
from ..flimds import UncorrectedFLIMds


class Scanner(Enum):
    PI_E710 = 1
    LSM = 3


class PTU(UncorrectedFLIMds):

    def __init__(self, file):
        self.file = file

        # Read header
        self.header, self.recstart = pq_header.read_header_ptu(self.file)
        self.nb_records = self.header[u'TTResult_NumberOfRecords']
        self.syncrate = self.header['TTResult_SyncRate']
        self.resolution = self.header['MeasDesc_Resolution']
        self.pixX = self.header['ImgHdr_PixX']
        self.pixY = self.header['ImgHdr_PixY']
        self.scanner = Scanner(self.header['ImgHdr_Ident'])

        if self.scanner == Scanner.PI_E710:
            raise NotImplementedError
        elif self.scanner == Scanner.LSM:
            self.lsm_frame = 0x1 << (self.header['ImgHdr_Frame'] - 1)
            self.lsm_line_start = 0x1 << (self.header['ImgHdr_LineStart'] - 1)
            self.lsm_line_stop = 0x1 << (self.header['ImgHdr_LineStop'] - 1)
        else:
            raise NotImplementedError

        self.binrep = 1 / (self.resolution * self.syncrate)

        # Load data
        self.x, self.y, self.f, self.d = self.interpret_records(*self.read_records())
        self.nb_of_photons = len(self.d)

    def read_records(self):
        channel, dtime, truetime = pq.read_records(self.file, self.nb_records,
                                                   self.recstart, self.syncrate, self.resolution)
        return channel, dtime, truetime

    def interpret_records(self, channel, dtime, truetime):
        if self.scanner == Scanner.PI_E710:
            x, y, f, d = pq.interpret_PI(channel, dtime,
                                         truetime, self.pixX, self.pixY, self.TStartTo,
                                         self.TStopTo,
                                         self.TStartFro, self.TStopFro, self.bidirect)

        elif self.scanner == Scanner.LSM:
            x, y, f, d = pq.interpret_LSM(channel, dtime,
                                          truetime, self.pixX, self.pixY, self.lsm_frame,
                                          self.lsm_line_start, self.lsm_line_stop)
            return x, y, f, d

    @property
    def frequency(self):
        return self.syncrate

    def histogram(self):
        return _histogram(self.d.max(), self.d, self.x, self.y, (self.pixX, self.pixY))

    def fourier_image(self, harm, binning=1):
        if binning != 1:
            return NotImplementedError
        else:
            return phasemod(self.d, self.x, self.y, self.d.max(), self.binrep, (self.pixX, self.pixY), harm=harm)


def phasemod(dtime, x, y, TACbins, TACrep, xydim, harm=0, image=True):
    """
    calculate complex coordinates

    Parameters
    -----------

    dtime: TAC bin (int16 array)

    x: x coordinates of photons (int16 array)

    y: y coordinates of photons (int16 array)

    TACbins: nb of bins with nonzero photons (dtime.max)

    TACrep: nb of bins corresponding to time between laser pulses
            (in theory should be the same as bins, but due to jitter in the laser
            stability some photons can arrive later)

    xydim: dimension of image

    harm: number of harmonics to calculate

    image: (bool) if True returns an image if False returns the average


    Returns
    -------

    im: image stack of dimension [n+1, ydimensions, xdimensions]
        or wave with dimensions [n+1]

    """
    binsX, binsY = xydim

    # create array with phase components
    e = np.ones(((TACbins + 1), (harm + 1)), dtype=np.complex64)
    xx = np.arange(TACbins + 1) * 2 * np.pi / TACrep
    for h in range(1, harm + 1):
        e[:, h] = np.exp(1j * xx * h)

    if image:
        im = np.zeros((harm + 1, binsY, binsX), dtype=np.complex64)
        _image(im, TACbins, e, dtime, x, y, xydim, harm=harm)
    else:
        hist = _histogram(TACbins, dtime, x, y, xydim).reshape(TACbins + 1, 1)
        e *= hist
        im = np.sum(e, axis=0)

    return im


@nb.njit
def _image(im, TACbins, e, dtime, x, y, xydim, harm=0):
    """
    calculate the complex fourier coefficients for every pixel in the image

    Parameters
    -----------

    im: image of shape [harm+1, ydimensions, xdimensions]

    e: complex wave of dimensions [TACbins+1, harm+1]

    dtime: TAC bin (int16 array)

    x: x coordinates of photons (int16 array)

    y: y coordinates of photons (int16 array)

    xydim: dimension of image

    harm: number of harmonics to calculate
    """
    binsX, binsY = xydim

    events = len(dtime)

    event_x = 0
    event_y = 0
    for event in range(events):
        event_x = x[event]
        event_y = y[event]
        event_t = dtime[event]
        for h in range(harm + 1):
            if event_x < 0 or event_x >= binsX or event_y < 0 or event_y >= binsY:
                continue
            if event_t > TACbins: continue
            im[h, event_y, event_x] += e[dtime[event], h]


@nb.njit
def _histogram(TACbins, dtime, x, y, xydim):
    """
    calculate the complex fourier coefficients for the whole measurement

    Parameters
    -----------

    im: complex array of len [harm+1]

    e: complex wave of dimensions [TACbins+1, harm+1]

    dtime: TAC bin (int16 array)

    x: x coordinates of photons (int16 array)

    y: y coordinates of photons (int16 array)

    xydim: dimension of image

    harm: number of harmonics to calculate
    """
    binsX, binsY = xydim
    events = len(dtime)
    event_x = 0
    event_y = 0

    hist = np.zeros(TACbins + 1, dtype=np.float32)
    for event in range(events):
        event_x = x[event]
        event_y = y[event]
        event_t = dtime[event]
        if event_x < 0 or event_x >= binsX or event_y < 0 or event_y >= binsY:
            continue
        if event_t > TACbins: continue
        hist[dtime[event]] += 1

    return hist
