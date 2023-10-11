"""
Core functionality for app.
"""
import numpy as np
import pandas as pd

from sources import get_gather
from filter import pass_filter


def generate_event_df(
        station_locations,
        angle_1=-90,
        angle_2=270,
        radius_1=2_000,
        radius_2=3_000,
        num_events=500,
):
    """Generate random event locations from centroid of stations"""
    centroid = np.mean(station_locations, axis=0)
    angles = np.linspace(angle_1, angle_2, num=num_events)
    # angles = np.sort(np.random.uniform(angle_1, angle_2, size=num_events))
    radii = np.random.uniform(radius_1, radius_2, size=num_events)
    y_values = np.sin(np.deg2rad(angles)) * radii
    x_values = np.cos(np.deg2rad(angles)) * radii
    # create data frame and return
    out = pd.DataFrame(index=range(num_events))
    out['x'] = centroid[0] + x_values
    out['y'] = centroid[1] + y_values
    out['phi'] = angles
    out['radius'] = radii
    return out


def get_duration(station_array, event_df, velocity):
    """Get required duration for gathers."""
    tt_min = np.inf
    tt_max = -np.inf

    for sta in station_array:
        dists = sta - event_df[['x', 'y']]
        tt = np.linalg.norm(dists, axis=1) / velocity

        tt_min = np.min([tt.min(), tt_min])
        tt_max = np.max([tt.max(), tt_max])

    return tt_min, tt_max


def create_gathers(
        station_array,
        event_df,
        source_type='ricker',
        velocity=2_000,
        dt=0.001,
        freq_min=None,
        freq_max=None,
        **kwargs,
):
    """Create gathers based on specified source time function."""
    out = []
    # get a nice plotting duration
    tt_min, tt_max = get_duration(station_array, event_df, velocity)
    tt_total = tt_max - tt_min
    t2 = tt_max + tt_total * 0.1
    t1 = tt_min - tt_total * 0.1

    for station in station_array:
        gather = get_gather(
            station,
            event_df,
            velocity=velocity,
            source_type=source_type,
            duration=t2,
            dt=dt,
            **kwargs,
        )
        filtered = pass_filter(gather, freq_min=freq_min, freq_max=freq_max)
        out.append(filtered.loc[slice(t1, t2)])
    return out


def correlate(gather_1, gather_2):
    """Correlate gathers together."""
    assert gather_1.shape == gather_2.shape
    shape = gather_1.shape

    # first pad each array with 0s
    padded_1 = np.zeros((shape[0] * 2 - 1, shape[1]), dtype=np.float64)
    padded_1[:shape[0], :] = gather_1.values
    padded_2 = np.zeros((shape[0] * 2 - 1, shape[1]), dtype=np.float64)
    padded_2[:shape[0], :] = gather_2.values

    # the transform to fourier domain
    fft1 = np.fft.rfft(padded_1, axis=0)
    fft2 = np.fft.rfft(padded_2, axis=0)

    # perform correlation in freq domain
    out_fft = fft1 * np.conj(fft2)

    out_time = np.fft.irfft(out_fft, axis=0)
    out = np.fft.fftshift(out_time, axes=0)

    time = gather_1.index.values
    duration = time.max() - time.min()
    time = np.linspace(-duration, duration, num=out.shape[0])
    return pd.DataFrame(out, index=time, columns=gather_1.columns)
