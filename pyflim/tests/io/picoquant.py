import json
import unittest

from pyflim.io.pq_header import read_header_ptu


class TestHeader(unittest.TestCase):

    def test_read_ptu_header(self):
        with open('tests/io/example.json') as file:
            expected_header, expected_records_start = json.load(file)

        header, records_start = read_header_ptu('tests/io/example.ptu')

        self.assertDictEqual(header, expected_header)
        self.assertEqual(records_start, expected_records_start)


if __name__ == '__main__':
    unittest.main()
