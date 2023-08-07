import numba as nb
import numpy as np


@nb.jit
def _bit_mask(length):
    return (1 << length) - 1


@nb.jit
def _bit_get(value, shift, length):
    return (value >> shift) & _bit_mask(length)


@nb.jit
def _bit_get_reverse(value, shift, length):
    #    return (value >> shift) & _bit_mask(length)

    tmp_val = (value >> shift) & _bit_mask(length)
    result = 0
    for n in range(length):
        result = (result << 1) + (tmp_val & 1)
        tmp_val >>= 1
    return result


@nb.njit
def _read_events(records, nb_records, syncrate, resolution):
    """
    read the BH records from an array-like object

    Parameters
    -----------

    records: an array like object over the uint32 records.

    nb_records: number of records in file.

    syncrate: synchronization rate in Hz.

    resolution: TAC resolution in s.

    Returns
    --------

    channel: int16 array

    dtime: int16 array

    truetime: double array
    """

    syncperiod = 1.0e9 / syncrate
    ofltime = np.uint64(0)
    truensync = 0.0
    event = 0

    channels = np.empty(nb_records, dtype=np.int16)
    dtimes = np.empty(nb_records, dtype=np.int16)
    truetimes = np.empty(nb_records, dtype=np.double)

    for n in range(nb_records):
        record = records[n]

        is_invalid = 0 != _bit_get(record, 31, 1)
        has_mtov = 0 != _bit_get(record, 30, 1)
        is_marker = 0 != _bit_get(record, 28, 1)

        # Markers are identified by channel numbers >= 32768
        if is_invalid:
            if not is_marker:
                if has_mtov:  # No event, just wraparound
                    ofltime += 0x1000 * (record & 0x0FFFFFFF)
                continue

        if has_mtov:
            ofltime += 0x1000

        truensync = 1.0 * ofltime + 1.0 * (record & 0x00000FFF)
        chan = _bit_get(record, 12, 4)

        if is_marker:
            dtimes[event] = 0
            chan += 0x8000
        else:
            dtimes[event] = 4096 - _bit_get(record, 16, 12)

        channels[event] = chan
        dtime = 4096 - _bit_get(record, 32 - 16, 12)
        truetimes[event] = truensync * syncperiod + dtime * resolution
        event += 1

    return channels[:event], dtimes[:event], truetimes[:event]


def read_records(fname_or_fp, nb_records, offset, syncrate, resolution):
    """
    read records of a pt3/ptu file

    Parameters
    ----------

    fname_or_fp: full path to pt3/ptu file or file-like object.

    nb_records: maximum number of records to be read.

    offset: number of bytes to skip until the start of the records.

    syncrate: synchronization rate in Hz.

    resolution: TAC resolution in s.


    Returns
    -------

    channel: int16 array

    dtime: int16 array

    truetime: double array
    """

    if isinstance(fname_or_fp, str):
        with open(fname_or_fp, mode="rb") as fp:
            read_records(fp, nb_records, offset, syncrate, resolution)

    fp = fname_or_fp

    records = np.memmap(fp, dtype="uint32", mode="r", offset=offset)
    channels, dtimes, truetimes = _read_events(
        records, nb_records, syncrate, resolution
    )

    return channels, dtimes, truetimes


@nb.njit
def interpret_AI(
    channel,
    dtime,
    truetime,
    pixX,
    pixY,
    lsm_frame,
    lsm_line_start,
    lsm_pixel_start,
    pixel_dwell_time,
):
    """
    calculate the x, y position and the frame from truetime

    Parameters
    ----------

    channel: int16 array

    dtime: int16 array

    truetime: double array

    pixX: pixels in x

    pixY: pixels in Y

    lsm_frame: marker for lsm frame

    lsm_line_start: marker for lsm line start

    lsm_line_stop: marker for lsm line stop

    x: empty numpy array

    y:

    Returns
    --------

    x: x coordinates of photons

    y: y coordinates of photons

    f: frame

    d: dtime
    """
    nb_events = len(channel)
    x = np.empty(nb_events, dtype=np.int16)
    y = np.empty(nb_events, dtype=np.int16)
    f = np.empty(nb_events, dtype=np.int16)

    # line_start = 0.0
    # line_time = 0.0

    n_pixels_per_line = -1
    n_lines_per_frame = -1

    # Calculate the number of pixels and lines in case it was not set

    # line_start = 0
    line = -1
    frame = -1
    x_pos = -1
    events_in_range = 0

    for i in range(nb_events):
        if channel[i] & 0x8000:
            if channel[i] & lsm_frame:
                if line >= 0:
                    if n_lines_per_frame >= 0:
                        if line != n_lines_per_frame - 1:
                            # This should not happen but we will not crash, rather either accept the
                            # missing pixels or extend the image for now

                            n_lines_per_frame = max(n_lines_per_frame, line + 1)
                            # raise Exception('Frames with different amount of lines found.')
                    else:
                        n_lines_per_frame = line + 1

                line = -1
                x_pos = -1  # Invalid until next line start
                frame += 1
            if channel[i] & lsm_line_start:
                if x_pos >= 0:
                    if n_pixels_per_line >= 0:
                        if x_pos != n_pixels_per_line:
                            # This should not happen but we will not crash, rather either accept the
                            # missing pixels or extend the image for now
                            # raise Exception('Lines with different amount of pixels found.')
                            n_pixels_per_line = max(n_pixels_per_line, x_pos)
                    else:
                        n_pixels_per_line = x_pos
                x_pos = 0
                line += 1
            if channel[i] & lsm_pixel_start:
                x_pos += 1

        elif frame >= 0 and line >= 0 and x_pos >= 0:
            # x_pos = (truetime[i] - line_start) / line_time * 1.0 * (pixX - 1)
            x[events_in_range] = np.int16(x_pos)
            y[events_in_range] = line
            f[events_in_range] = frame
            dtime[events_in_range] = dtime[i]
            events_in_range += 1

    if n_lines_per_frame < 0:
        # Only one frame in file
        n_lines_per_frame = line + 1

    return (
        x[:events_in_range],
        y[:events_in_range],
        f[:events_in_range],
        dtime[:events_in_range],
        n_pixels_per_line,
        n_lines_per_frame,
    )
