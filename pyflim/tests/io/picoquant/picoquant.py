import json
import pathlib
import unittest

import numpy as np

from pyflim.io.picoquant import pq_header, pq_numba


class TestHeader(unittest.TestCase):
    ptu_file = pathlib.Path('tests/io/picoquant/ptu_example.ptu')
    pt3_file = pathlib.Path('tests/io/picoquant/pt3_example.pt3')

    def test_read_ptu_header(self):
        with open(self.ptu_file.with_suffix('.json')) as file:
            expected_header, expected_records_start = json.load(file)

        header, records_start = pq_header.read_header_ptu(self.ptu_file)

        self.assertDictEqual(header, expected_header)
        self.assertEqual(records_start, expected_records_start)

    def test_read_pt3_header(self):
        with open(self.pt3_file.with_suffix('.json')) as file:
            expected_header, expected_records_start = json.load(file)

        header, records_start = pq_header.read_header_pt3(self.pt3_file)
        for key, value in header.items():
            if isinstance(value, np.ndarray):
                header[key] = value.tolist()

        self.assertDictEqual(header, expected_header)
        self.assertEqual(records_start, expected_records_start)


class TestReader(unittest.TestCase):
    def setUp(self):
        self.ptu_file = pathlib.Path('tests/io/picoquant/ptu_example.ptu')

        self.header, self.records_start = pq_header.read_header_ptu(self.ptu_file)
        self.num_records = self.header['TTResult_NumberOfRecords']
        self.syncrate = self.header['TTResult_SyncRate']
        self.resolution = self.header['MeasDesc_Resolution']

        data = np.load(self.ptu_file.with_suffix('.npz'))
        self.channels, self.dtimes, self.truetimes = data['read_records']
        self.x, self.y, self.f, self.d = data['interpret_records']

    def test_read_records(self):
        channels, dtimes, truetimes = pq_numba.read_records(self.ptu_file,
                                                            self.num_records,
                                                            self.records_start,
                                                            self.syncrate,
                                                            self.resolution)

        self.assertTrue(np.all(channels == self.channels))
        self.assertTrue(np.all(dtimes == self.dtimes))
        self.assertTrue(np.all(truetimes == self.truetimes))

    def test_interpret_LSM_records(self):
        pixX = self.header['ImgHdr_PixX']
        pixY = self.header['ImgHdr_PixY']
        lsm_frame = 0x1 << (self.header['ImgHdr_Frame'] - 1)
        lsm_line_start = 0x1 << (self.header['ImgHdr_LineStart'] - 1)
        lsm_line_stop = 0x1 << (self.header['ImgHdr_LineStop'] - 1)
        x, y, f, d = pq_numba.interpret_LSM(self.channels, self.dtimes, self.truetimes,
                                            pixX, pixY, lsm_frame, lsm_line_start, lsm_line_stop)

        self.assertTrue(np.all(x == self.x))
        self.assertTrue(np.all(y == self.y))
        self.assertTrue(np.all(f == self.f))
        self.assertTrue(np.all(d == self.d))


if __name__ == '__main__':
    unittest.main()
