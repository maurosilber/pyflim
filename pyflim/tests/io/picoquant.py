import json
import unittest

import numpy as np

from pyflim.io.pq_header import read_header_ptu, read_header_pt3


class TestHeader(unittest.TestCase):

    def test_read_ptu_header(self):
        with open('tests/io/ptu_example.json') as file:
            expected_header, expected_records_start = json.load(file)

        header, records_start = read_header_ptu('tests/io/ptu_example.ptu')

        self.assertDictEqual(header, expected_header)
        self.assertEqual(records_start, expected_records_start)

    def test_read_pt3_header(self):
        with open('tests/io/pt3_example.json') as file:
            expected_header, expected_records_start = json.load(file)

        header, records_start = read_header_pt3('tests/io/pt3_example.pt3')
        for key, value in header.items():
            if isinstance(value, np.ndarray):
                header[key] = value.tolist()

        self.assertDictEqual(header, expected_header)
        self.assertEqual(records_start, expected_records_start)


if __name__ == '__main__':
    unittest.main()
