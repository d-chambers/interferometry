"""
Module for generating different sources
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUPPORTED_SOURCE_TYPES = {"ricker", "random"}


def _translate_times(time, pressure, travel_times, duration, dt):
    """Broadcast and expand time and pressure record."""
    if len(time) != len(travel_times):  # need to broadcast
        trace_times = time[None, :] + travel_times[:, None]
        gt_0_ind = np.argmax(trace_times > 0, axis=1).astype(np.int64)
        total_samples = int(duration / dt)
        data = np.stack([pressure[x: x + total_samples] for x in gt_0_ind], axis=-1)
        time = np.arange(start=0, stop=duration, step=dt)
        ser = pd.DataFrame(data=data, index=time)
    else:
        raise NotImplementedError("only translation times for now")
    return ser


def get_gather(station, event_df, duration, dt, velocity, source_type, **kwargs):
    """Get a single dataframe gather."""
    event_array = event_df[['x', 'y']].values
    dists = np.linalg.norm(event_array - station, axis=1)
    travel_times = dists / velocity
    if source_type.lower() not in SUPPORTED_SOURCE_TYPES:
        msg = f"Only source types supported are: {SUPPORTED_SOURCE_TYPES}"
        raise ValueError(msg)

    if source_type == "ricker":
        time, pressure = ricker(duration=3 * duration, dt=dt)
    else:
        raise NotImplementedError("net yet!")
    df = _translate_times(
        time, pressure, travel_times, duration=duration, dt=dt
    )
    df.columns = event_df['phi']
    df.index.name = 'time'
    df.columns.name = "angle"
    return df


def ricker(f=30, duration=1.0, dt=0.001):
    """Ricker wavelet, taken from https://subsurfwiki.org/wiki/Ricker_wavelet"""
    time = np.arange(-duration / 2, (duration - dt) / 2, dt)
    exp_term = np.exp(-(np.pi ** 2) * (f ** 2) * (time ** 2))
    amp_term = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (time ** 2))
    out = amp_term * exp_term
    return time, out


if __name__ == "__main__":
    t, w = ricker()
    plt.plot(t, w)
    plt.show()
