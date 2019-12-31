import struct
from enum import IntFlag

import numpy as np


class HeaderTypes(IntFlag):
    """Format types in header file"""
    empty8 = 0xFFFF0008
    bool8 = 0x00000008
    int8 = 0x10000008
    bit_set64 = 0x11000008
    color8 = 0x12000008
    float8 = 0x20000008
    datetime = 0x21000008
    float8_array = 0x2001FFFF
    ANSI_string = 0x4001FFFF
    wide_string = 0x4002FFFF
    binary_blob = 0xFFFFFFFF


def read_header_ptu(path):
    """
    Read header of a .ptu file.
    
    Parameters
    ----------
    path : string or pathlib.Path
        The file to read.
    
    Returns
    -------
    header : dict
        The header of the ptu file.
    records_start : int
        File location where the TTTR information starts.
    """

    def decode(s):
        return s.decode('utf-8').rstrip('\x00')

    with open(path, 'rb') as file:
        s = file.read(16)

        if decode(s[:8]) != 'PQTTTR':
            raise ValueError("Not a .ptu file.")

        header = {'Version': decode(s[8:])}
        while True:
            s = file.read(48)
            tag_id = decode(s[:32])
            tag_idx, tag_type_code = struct.unpack('<iI', s[32:40])
            tag_name = tag_id + str(tag_idx) if tag_idx > -1 else tag_id

            if tag_type_code in (HeaderTypes.empty8, HeaderTypes.int8, HeaderTypes.bit_set64, HeaderTypes.color8):
                tag_int = struct.unpack('<q', s[40:])[0]
                header[tag_name] = tag_int

            elif tag_type_code == HeaderTypes.bool8:
                tag_int = struct.unpack('<q', s[40:])[0]
                header[tag_name] = bool(tag_int)

            elif tag_type_code == HeaderTypes.float8:
                tag_int = struct.unpack('<d', s[40:])[0]
                header[tag_name] = tag_int

            elif tag_type_code == HeaderTypes.float8_array:
                tag_int = struct.unpack('<q', s[40:])[0]
                ss = file.read(tag_int)
                header[tag_name] = struct.unpack('<' + (tag_int / 8) * 'd', ss)

            elif tag_type_code == HeaderTypes.datetime:
                tag_int = struct.unpack('<d', s[40:])[0]
                header[tag_name] = tag_int

            elif tag_type_code in (HeaderTypes.ANSI_string, HeaderTypes.wide_string):
                tag_int = struct.unpack('<q', s[40:])[0]
                ss = file.read(tag_int)
                if tag_name in ('$Comment', 'File_Comment'): continue
                header[tag_name] = decode(ss)

            elif tag_type_code == HeaderTypes.binary_blob:
                tag_int = struct.unpack('<q', s[40:])[0]
                ss = file.read(tag_int)
                header[tag_name] = ss
            else:
                raise

            if tag_id == 'Header_End':
                break

        records_start = file.tell()
    return header, records_start


def read_header_pt3(path):
    """
    Read header of a .pt3 file.

    Parameters
    ----------
    path : string or pathlib.Path
        The file to read.

    Returns
    -------
    header : dict
        The header of the pt3 file.
    records_start : int
        File location where the TTTR information starts.
    """

    def decode(s):
        return s.decode('utf-8').rstrip('\x00')

    with open(path, 'rb') as file:
        # ASCII header
        s = file.read(328)
        header = {'Ident': decode(s[:16]),
                  'FormatVersion': decode(s[16:22]),
                  'CreatorName': decode(s[22:40]),
                  'CreatorVersion': decode(s[40:52]),
                  'FileTime': decode(s[52:70]),
                  'CRFL': decode(s[70:72]),
                  'CommentField': decode(s[72:])}

        # binary header
        DISPCURVES = 8
        s = file.read(72)
        # Tacq in ms
        # DispTime in ns
        for key, value in zip(('Curves', 'BitsPerRecord', 'RoutingChannels', 'NumberOfBoards',
                               'ActiveCurve', 'MeasMode', 'SubMode', 'RangeNo', 'Offset', 'Tacq',
                               'StopAt', 'StopOnOvfl', 'Restart', 'DispLinLog', 'DispTimeFrom',
                               'DispTimeTo', 'DispCountsFrom', 'DispCountsTo'),
                              struct.unpack('<' + 18 * 'I', s)):
            header[key] = value

        s = file.read(DISPCURVES * 8)
        header['DispCurves'] = np.array(struct.unpack('<' + DISPCURVES * 'II', s)).reshape(2, DISPCURVES)
        s = file.read(12 * 3)
        header['Params'] = np.array(struct.unpack('<fffffffff', s)).reshape(3, 3)
        s = file.read(36)
        for key, value in zip(('RepeatMode', 'RepeatsPerCurve', 'RepeatTime', 'RepeatWaitTime'),
                              struct.unpack('<' + 4 * 'I', s[:16])):
            header[key] = value
        header['ScriptName'] = decode(s[16:])

        # board specific header
        s = file.read(24)
        header['HardwareIdent'] = decode(s[:16])
        header['HardwareVersion'] = decode(s[16:])

        s = file.read(33 * 4)
        for key, value in zip(('HardwareSerial', 'SyncDivider', 'CFDZeroCross0', 'CFDLevel0',
                               'CFDZeroCross1', 'CFDLevel1', 'Resolution', 'RouterModelCode',
                               'RouterEnabled', 'RtChan1_InputType', 'RtChan1_InputLevel',
                               'RtChan1_InputEdge', 'RtChan1_CFDPresent', 'RtChan1_CFDLevel',
                               'RtChan1_CFDZeroCross', 'RtChan2_InputType', 'RtChan2_InputLevel',
                               'RtChan2_InputEdge', 'RtChan2_CFDPresent', 'RtChan2_CFDLevel',
                               'RtChan2_CFDZeroCross', 'RtChan3_InputType', ' RtChan3_InputLevel',
                               'RtChan3_InputEdge', 'RtChan3_CFDPresent', 'RtChan3_CFDLevel',
                               'RtChan3_CFDZeroCross', 'RtChan4_InputType', ' RtChan4_InputLevel',
                               'RtChan4_InputEdge', 'RtChan4_CFDPresent', 'RtChan4_CFDLevel',
                               'RtChan4_CFDZeroCross'),
                              struct.unpack('<' + 6 * 'I' + 'f' + 26 * 'I', s)):
            header[key] = value

        # TTTR mode specific header
        s = file.read(9 * 4)
        for key, value in zip(('ExtDevices', 'Reserved1', 'Reserved2', 'CntRate0', 'CntRate1',
                               'StopAfter', 'StopReason', 'Records', 'SpecHeaderLength'),
                              struct.unpack('<' + 9 * 'I', s)):
            header[key] = value

        # Imaging Header
        s = file.read(8)
        header['dimensions'], header['Ident'] = struct.unpack('<II', s)

        if header['Ident'] == 1:
            header['ScannerType'] = 'PI E710'
            h = ('TimerPerPixel', 'Acceleration', 'Pattern',
                 'Reserved', 'X0', 'Y0', 'PixX', 'PixY', 'PixResol', 'TStartTo',
                 'TStopTo', 'TStartFro', 'TStopFro')
            s = file.read(13 * 4)

            for i, value in enumerate(struct.unpack('<IIIIffIIfffff', s)):
                header[h[i]] = value

        if header['Ident'] == 4:
            header['ScannerType'] = 'KDT180-100-lm'
            h = ('Velocity', 'Acceleration', 'Pattern', 'Reserved', 'X0', 'Y0',
                 'PixX', 'PixY', 'PixResol')
            s = file.read(9 * 4)
            for i, value in enumerate(struct.unpack('<IIIIIIffIIf', s)):
                header[h[i]] = value

        if header['Ident'] == 3:
            header['ScannerType'] = 'LSM'
            h = ('Frame', 'LineStart', 'LineStop', 'Pattern', 'PixX', 'PixY')
            s = file.read(6 * 4)
            for i, value in enumerate(struct.unpack('<IIIIII', s)):
                header[h[i]] = value

        records_start = file.tell()
    return header, records_start
