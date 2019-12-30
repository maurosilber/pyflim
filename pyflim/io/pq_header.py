import struct

import numpy as np

# format types in header file
tyEmpty8 = 0xFFFF0008
tyBool8 = 0x00000008
tyInt8 = 0x10000008
tyBitSet64 = 0x11000008
tyColor8 = 0x12000008
tyFloat8 = 0x20000008
tyTDateTime = 0x21000008
tyFloat8Array = 0x2001FFFF
tyAnsiString = 0x4001FFFF
tyWideString = 0x4002FFFF
tyBinaryBlob = 0xFFFFFFFF


def read_header_ptu(fname):
    """
    load the header file of a .ptu file
    
    Parameters
    ----------
    
    fname: full path to .ptu file
    
    
    Returns
    --------
    
    header: header of the ptu file
    
    rec_start: file location where the TTTR information starts    
    """

    f = open(fname, "rb")
    s = f.read(16)
    if s[:8].decode("utf-8").rstrip("\x00") != 'PQTTTR':
        return None

    header = dict()
    header["Version"] = s[8:].decode("utf-8").rstrip("\x00")

    while True:

        s = f.read(48)
        TagId = s[:32].decode("utf-8").rstrip("\x00")
        TagIdx, TagTypeCode = struct.unpack("<iI", s[32:40])

        if TagIdx > -1:
            TagName = TagId + str(TagIdx)
        else:
            TagName = TagId

        if TagTypeCode in (tyEmpty8, tyInt8, tyBitSet64, tyColor8):
            TagInt = struct.unpack("<q", s[40:])[0]
            header[TagName] = TagInt

        elif TagTypeCode == tyBool8:
            TagInt = struct.unpack("<q", s[40:])[0]
            header[TagName] = bool(TagInt)

        elif TagTypeCode == tyFloat8:
            TagInt = struct.unpack("<d", s[40:])[0]
            header[TagName] = TagInt

        elif TagTypeCode == tyFloat8Array:
            TagInt = struct.unpack("<q", s[40:])[0]
            ss = f.read(TagInt)
            header[TagName] = struct.unpack("<" + (TagInt / 8) * "d", ss)

        elif TagTypeCode == tyTDateTime:
            TagInt = struct.unpack("<d", s[40:])[0]
            header[TagName] = TagInt

        elif TagTypeCode in (tyAnsiString, tyWideString):
            TagInt = struct.unpack("<q", s[40:])[0]
            ss = f.read(TagInt)
            if TagName in ("$Comment", "File_Comment"): continue
            header[TagName] = ss.decode("utf-8").rstrip("\x00")

        elif TagTypeCode == tyBinaryBlob:
            TagInt = struct.unpack("<q", s[40:])[0]
            ss = f.read(TagInt)
            header[TagName] = ss
        else:
            raise

        if TagId == 'Header_End':
            break

    rec_start = f.tell()
    f.close()

    return header, rec_start


def read_header_pt3(fname):
    """
    load the header file of a .pt3 file
    
    Parameters
    ----------
    
    fname: full path to .pt3 file
    
    
    Returns
    --------
    
    header: header of the pt3 file
    
    rec_start: file location where the TTTR information starts    
    """

    f = open(fname, "rb")

    # ASCII header
    s = f.read(328)
    header = dict()
    header["Ident"] = s[:16].decode("utf-8").rstrip("\x00")
    header["FormatVersion"] = s[16:22].decode("utf-8").rstrip("\x00")
    header["CreatorName"] = s[22:40].decode("utf-8").rstrip("\x00")
    header["CreatorVersion"] = s[40:52].decode("utf-8").rstrip("\x00")
    header["FileTime"] = s[52:70].decode("utf-8").rstrip("\x00")
    header["CRFL"] = s[70:72].decode("utf-8").rstrip("\x00")
    header["CommentField"] = s[72:].decode("utf-8").rstrip("\x00")

    # binary header
    DISPCURVES = 8
    s = f.read(72)
    h = ["Curves", "BitsPerRecord", "RoutingChannels", "NumberOfBoards",
         "ActiveCurve", "MeasMode", "SubMode", "RangeNo", "Offset", "Tacq",
         "StopAt", "StopOnOvfl", "Restart", "DispLinLog", "DispTimeFrom",
         "DispTimeTo", "DispCountsFrom", "DispCountsTo"]
    # Tacq in ms
    # DispTime in ns
    for i, val in enumerate(struct.unpack("<" + 18 * "I", s)):
        header[h[i]] = val

    s = f.read(DISPCURVES * 8)
    header["DispCurves"] = np.array(struct.unpack("<" + DISPCURVES * "II", s)).reshape(2, DISPCURVES)
    s = f.read(12 * 3)
    header["Params"] = np.array(struct.unpack("<fffffffff", s)).reshape(3, 3)
    s = f.read(36)
    h = ["RepeatMode", "RepeatsPerCurve", "RepeatTime", "RepeatWaitTime"]
    for i, val in enumerate(struct.unpack("<" + 4 * "I", s[:16])):
        header[h[i]] = val
    header["ScriptName"] = s[16:].decode("utf-8").rstrip("\x00")

    # board specific header
    s = f.read(24)
    header["HardwareIdent"] = s[:16].decode("utf-8").rstrip("\x00")
    header["HardwareVersion"] = s[16:].decode("utf-8").rstrip("\x00")
    h = ["HardwareSerial", "SyncDivider", "CFDZeroCross0", "CFDLevel0",
         "CFDZeroCross1", "CFDLevel1", "Resolution", "RouterModelCode",
         "RouterEnabled", "RtChan1_InputType", "RtChan1_InputLevel",
         "RtChan1_InputEdge", "RtChan1_CFDPresent", "RtChan1_CFDLevel",
         "RtChan1_CFDZeroCross", "RtChan2_InputType", "RtChan2_InputLevel",
         "RtChan2_InputEdge", "RtChan2_CFDPresent", "RtChan2_CFDLevel",
         "RtChan2_CFDZeroCross", "RtChan3_InputType", " RtChan3_InputLevel",
         "RtChan3_InputEdge", "RtChan3_CFDPresent", "RtChan3_CFDLevel",
         "RtChan3_CFDZeroCross", "RtChan4_InputType", " RtChan4_InputLevel",
         "RtChan4_InputEdge", "RtChan4_CFDPresent", "RtChan4_CFDLevel",
         "RtChan4_CFDZeroCross"]

    s = f.read(33 * 4)
    for i, val in enumerate(struct.unpack("<" + 6 * "I" + "f" + 26 * "I", s)):
        header[h[i]] = val

        # TTTR mode specific header
    h = ["ExtDevices", "Reserved1", "Reserved2", "CntRate0", "CntRate1",
         "StopAfter", "StopReason", "Records", "SpecHeaderLength"]
    s = f.read(9 * 4)
    for i, val in enumerate(struct.unpack("<" + 9 * "I", s)):
        header[h[i]] = val

        # Imaging Header

    s = f.read(8)
    header["dimensions"], header["Ident"] = struct.unpack("<II", s)

    if header["Ident"] == 1:
        header["ScannerType"] = "PI E710"
        h = ["TimerPerPixel", "Acceleration", "Pattern",
             "Reserved", "X0", "Y0", "PixX", "PixY", "PixResol", "TStartTo",
             "TStopTo", "TStartFro", "TStopFro"]
        s = f.read(13 * 4)

        for i, val in enumerate(struct.unpack("<IIIIffIIfffff", s)):
            header[h[i]] = val

    if header["Ident"] == 4:
        header["ScannerType"] = "KDT180-100-lm"
        h = ["Velocity", "Acceleration", "Pattern", "Reserved", "X0", "Y0",
             "PixX", "PixY", "PixResol"]
        s = f.read(9 * 4)
        for i, val in enumerate(struct.unpack("<IIIIIIffIIf", s)):
            header[h[i]] = val

    if header["Ident"] == 3:
        header["ScannerType"] = "LSM"
        h = ["Frame", "LineStart", "LineStop", "Pattern", "PixX", "PixY"]
        s = f.read(6 * 4)
        for i, val in enumerate(struct.unpack("<IIIIII", s)):
            header[h[i]] = val

    rec_start = f.tell()
    return header, rec_start
