from enum import Enum

from . import pq_header, pq_numba as pq
from ..functions import phasemod, _histogram
from ...flimds import UncorrectedFLIMds


class Scanner(Enum):
    PI_E710 = 1
    LSM = 3


class PTU(UncorrectedFLIMds):

    def __init__(self, filename):
        self.file = filename

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
