import os
import struct


def read_header_spc(fname):
    """
    load the header file of an .spc file

    Parameters
    ----------

    fname: full path to .spc file


    Returns
    --------

    header: header of the ptu file

    rec_start: file location where the TTTR information starts
    """

    f = open(fname, "rb")
    h = struct.unpack("i", f.read(4))[0]

    # There is one 32-bit integer as header with the following structure:
    # byte 0, 1, 2 = macro time clock in 0.1 ns units(for 50ns value 500 is set)
    # byte 3 : bit 7 = 1 (Invalid photon)
    # byte 3 : bits 3 - 6 = number of routing bits,
    # Currently always 4 channels even though not all detectors may be active
    # byte 3 : bits 0 - 2 reserved

    # Sanity check

    if not (h & 0x80000000) == 0x80000000:
        raise Exception(
            "Invalid .spc file: Header photon frame does not have invalid photon bit set."
        )

    header = dict()
    header["macro_clock"] = 0.1e-9 * (h & 0x00FFFFFF)
    n = (h & (0xF << 27)) >> 27

    # There was a bug in imspector that shifted the number of routing bits to position 28
    # instead for the longest time. This version ALSO always wrote a 2 in there, so if we
    # get 4 routing bits (would correspond to 15 channels, we will write 2 routing bits for
    # now, which used to be the imspector standard.
    if n == 4:
        n = 2

    statinfo = os.stat(fname)
    header["num_records"] = int(statinfo.st_size / 4)

    # rec_start is always afer 4 bytes
    rec_start = 4
    return header, rec_start
