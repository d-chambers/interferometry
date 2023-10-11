"""
Module for Butterworth bandpass filter on array.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import median_filter as nd_median_filter
from scipy.signal import iirfilter, sosfilt, sosfiltfilt, zpk2sos



def _check_filter_range(nyquist, low, high, filt_min, filt_max):
    """Simple check on filter parameters."""
    # ensure filter bounds are within nyquist
    if low is not None and ((0 > low) or (low > 1)):
        msg = f"possible filter bounds are [0, {nyquist}] you passed {filt_min}"
        raise ValueError(msg)
    if high is not None and ((0 > high) or (high > 1)):
        msg = f"possible filter bounds are [0, {nyquist}] you passed {filt_max}"
        raise ValueError(msg)
    if high is not None and low is not None and high <= low:
        msg = (
            "Low filter param must be less than high filter param, you passed:"
            f"filt_min = {filt_min}, filt_max = {filt_max}"
        )
        raise ValueError(msg)


def _get_sos(sr, filt_min, filt_max, corners):
    """Get second order sections from sampling rate and filter bounds."""
    nyquist = 0.5 * sr
    low = None if pd.isnull(filt_min) else filt_min / nyquist
    high = None if pd.isnull(filt_max) else filt_max / nyquist
    _check_filter_range(nyquist, low, high, filt_min, filt_max)

    if (low is not None) and (high is not None):  # apply bandpass
        z, p, k = iirfilter(
            corners, [low, high], btype="band", ftype="butter", output="zpk"
        )
    elif low is not None:
        z, p, k = iirfilter(
            corners, low, btype="highpass", ftype="butter", output="zpk"
        )
    else:
        assert high is not None
        z, p, k = iirfilter(
            corners, high, btype="lowpass", ftype="butter", output="zpk"
        )
    return zpk2sos(z, p, k)


def pass_filter(df, freq_min=None, freq_max=None, corners=4, zerophase=True, ):
    """
    Apply a Butterworth pass filter (bandpass, highpass, or lowpass).

    Parameters
    ----------
    df
        A data array with time-series data.
    freq_min
        Minimum freq, if blank perform highpass
    freq_max
        Maximum freq, if blank do lowpass
    corners
        Number of corners in filter.
    zerophase
        If True, perform filter twice to avoid phase-shift.
    """
    if freq_min is None and freq_max is None:
        return df
    assert df.index.name == "time"
    data = df.values
    times = df.index.values
    sr = np.median((times[1:] - times[:-1]))
    # get nyquist and low/high in terms of nyquist
    sos = _get_sos(sr, freq_min, freq_max, corners)
    if zerophase:
        out = sosfiltfilt(sos, data, axis=0)
    else:
        out = sosfilt(sos, data, axis=0)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


