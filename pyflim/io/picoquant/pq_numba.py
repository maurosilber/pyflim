import numba as nb
import numpy as np

T3_WRAP_AROUND = 65536


@nb.jit
def _bit_mask(length):
    return (1 << length) - 1


@nb.jit
def _bit_get(value, shift, length):
    return (value >> shift) & _bit_mask(length)


@nb.njit
def _read_events(records, num_records, syncrate, resolution):
    """
    Read the TTTR data from an array-like object
    
    Parameters
    -----------
    records: array_like
        An array-like object over the uint32 records.
    num_records: int
        Number of records in file.
    syncrate: int
        Synchronization rate in Hz.
    resolution: int
        TAC resolution in s.
    
    Returns
    --------
    channel : int16 array
    dtime : int16 array
    truetime : double array
    """

    syncperiod = 1.0e9 / syncrate
    ofltime = np.uint64(0)
    truensync = 0.0
    event = 0

    channels = np.empty(num_records, dtype=np.int16)
    dtimes = np.empty(num_records, dtype=np.int16)
    truetimes = np.empty(num_records, dtype=np.double)

    for n in range(num_records):
        record = records[n]  # all 32 bits
        nsync = _bit_get(record, 0, 16)  # lowest 16 bits
        channel = _bit_get(record, 32 - 4, 4)  # upper 4 bits

        if channel == 15:
            markers = _bit_get(record, 32 - 16, 4)  # lowest 4 bits of the upper 16 bits
            if markers == 0:  # Overflow
                ofltime = ofltime + T3_WRAP_AROUND
                continue

        dtime = _bit_get(record, 32 - 16, 12)
        if 1 <= channel <= 4 or channel == 15:  # Photon or spatial marker event
            truensync = 1.0 * ofltime + 1.0 * nsync
            channels[event] = channel
            dtimes[event] = dtime
            truetimes[event] = truensync * syncperiod + dtime * resolution
            event += 1

    return channels[:event], dtimes[:event], truetimes[:event]


def read_records(file, num_records, offset, syncrate, resolution):
    """Read records of a pt3/ptu file.

    Parameters
    ----------
    file : os.PathLike
    num_records : int
        Maximum number of records to be read.
    offset : int
        Number of bytes to skip until the start of the records.
    syncrate : int
        Synchronization rate in Hz.
    resolution : int
        TAC resolution in s.

    Returns
    -------
    channel: int16 array
    dtime: int16 array
    truetime: double array
    """

    records = np.memmap(file, dtype='uint32', mode='r', offset=offset)
    channels, dtimes, truetimes = _read_events(records, num_records, syncrate, resolution)

    return channels, dtimes, truetimes


@nb.njit
def interpret_LSM(channel, dtime, truetime, pixX, pixY, lsm_frame,
                  lsm_line_start, lsm_line_stop):
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
    
    dtime: dtime
    """
    nb_events = len(channel)
    x = np.empty(nb_events, dtype=np.int16)
    y = np.empty(nb_events, dtype=np.int16)
    f = np.empty(nb_events, dtype=np.int16)

    line_start = 0.0
    line_time = 0.0
    lines = 0

    # calculate the dwell time (time spent per line)
    for i in range(nb_events):
        if channel[i] == 15:
            if dtime[i] == lsm_line_start:
                line_start = truetime[i]
            elif dtime[i] == lsm_line_stop:
                line_time += truetime[i] - line_start
                lines += 1
    line_time /= 1.0 * lines

    line_start = 0
    line = -1
    frame = -1
    x_pos = 0.0
    line_started = False
    events_in_range = 0

    for i in range(nb_events):
        if channel[i] == 15:
            if dtime[i] == lsm_frame:
                line = -1
                frame += 1

            elif dtime[i] == lsm_line_start:
                line_started = True
                line_start = truetime[i]
                line += 1
                if line >= pixY:
                    line = 0
            elif dtime[i] == lsm_line_stop:
                line_started = False
        elif line_started == False:
            continue
        elif channel[i] > 0:
            x_pos = (truetime[i] - line_start) / line_time * 1.0 * (pixX - 1)
            x[events_in_range] = np.int16(x_pos)
            y[events_in_range] = line
            f[events_in_range] = frame
            dtime[events_in_range] = dtime[i]
            events_in_range += 1

    return x[:events_in_range], y[:events_in_range], f[:events_in_range], dtime[:events_in_range]


@nb.njit
def interpret_PI(channel, dtime, truetime, pixX, pixY, t_start_to, t_stop_to,
                 t_start_fro, t_stop_fro, bidirect=False):
    """
    calculate the x, y position and the frame from truetime
    
    Parameters
    ----------
    
    channel: int16 array
    
    dtime: int16 array
    
    truetime: double array
    
    pixX: pixels in x
    
    pixY: pixels in Y
    
    t_start_to: delay of start marker 
    
    t_stop_to: delay of stop marker
    
    t_start_fro: delay of start marker (coming back - bidirectional)
    
    t_stop_fro: delay of stop marker (coming back - bidirectional)
    
    bidirect: bidirectional scanning, defaults to "False"
    
    Returns
    --------
    
    x: x coordinates of photons
    
    y: y coordinates of photons
    
    f: frame
    
    dtime: dtime
    """

    nb_events = len(channel)

    x = np.zeros(nb_events, dtype=np.int16)
    y = np.zeros(nb_events, dtype=np.int16)
    f = np.zeros(nb_events, dtype=np.int16)

    line_start = -1.0
    line_time = 0.0
    lines = 0
    scanning_to = True

    # calculate the dwell time (time spent per line)
    for i in range(nb_events):
        if channel[i] == 15:
            if line_start < 0:
                line_start = truetime[i]
            else:
                if scanning_to:
                    line_time += truetime[i] - line_start
                    line_start = truetime[i]
                    lines += 1
                else:
                    line_start = -1
                if bidirect:
                    scanning_to = not scanning_to
    line_time /= (1.0 * lines)
    offset_to = 1.0 * t_start_to * line_time
    offset_fro = 1.0 * t_start_fro * line_time
    dwell_t_to = 1.0 * (t_stop_to - t_start_to) * line_time
    dwell_t_fro = 1.0 * (t_stop_fro - t_start_fro) * line_time

    # interpret the markers and calculate the x, y coordinates
    line_start = 0
    line = -1
    frame = 0
    x_pos = 0.0
    scanning_to = True
    events_in_range = 0

    for i in range(nb_events):
        if channel[i] == 15:
            line += 1
            if line >= pixY:
                line = 0
                frame += 1
            if line > 0 and bidirect:
                scanning_to = not scanning_to
            if scanning_to:
                line_start = truetime[i]  # reverse scanning uses the previous trigger signal

        elif channel[i] > 0:
            if scanning_to:
                x_pos = (truetime[i] - line_start - offset_to) / dwell_t_to * pixX
            else:
                x_pos = (truetime[i] - line_start - offset_fro) / dwell_t_fro * pixX
                x_pos = pixX - x_pos - 1
            if x_pos > 0 and x_pos < pixX:
                x[events_in_range] = x_pos
                y[events_in_range] = line
                f[events_in_range] = frame
                dtime[events_in_range] = dtime[i]
                events_in_range += 1

    return x[:events_in_range], y[:events_in_range], f[:events_in_range], dtime[:events_in_range]
