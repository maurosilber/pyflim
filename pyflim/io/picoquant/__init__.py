from enum import Enum

from . import pq_header, pq_numba as pq
from ..functions import histogram, fourier_image
from ...flimds import UncorrectedFLIMds


class Scanner(Enum):
    PI_E710 = 1
    LSM = 3


class PTU(UncorrectedFLIMds):

    def __init__(self, filename):
        self.file = filename

        # Read header
        self.header, self.records_start = pq_header.read_header_ptu(self.file)
        self.num_records = self.header[u'TTResult_NumberOfRecords']
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

        self.TAC_period = 1 / (self.resolution * self.syncrate)

        # Load data
        self.x, self.y, self.f, self.dtime = self._interpret_records(*self._read_records())
        self.num_TAC_bins = self.dtime.max() + 1

    def _read_records(self):
        channel, dtime, truetime = pq.read_records(self.file, self.num_records, self.records_start,
                                                   self.syncrate, self.resolution)
        return channel, dtime, truetime

    def _interpret_records(self, channel, dtime, truetime):
        if self.scanner == Scanner.PI_E710:
            x, y, f, d = pq.interpret_PI(channel, dtime, truetime, self.pixX, self.pixY,
                                         self.TStartTo, self.TStopTo, self.TStartFro, self.TStopFro, self.bidirect)

        elif self.scanner == Scanner.LSM:
            x, y, f, d = pq.interpret_LSM(channel, dtime, truetime, self.pixX, self.pixY,
                                          self.lsm_frame, self.lsm_line_start, self.lsm_line_stop)
            return x, y, f, d

    @property
    def frequency(self):
        return self.syncrate

    def histogram(self, mask=None):
        return histogram(self.dtime, self.x, self.y, self.num_TAC_bins, mask=mask)

    def fourier_image(self, harmonics, mask=None):
        return fourier_image((self.pixY, self.pixX), harmonics, self.dtime, self.x, self.y,
                             self.num_TAC_bins, self.TAC_period, mask=mask)
