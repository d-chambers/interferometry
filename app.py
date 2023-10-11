"""
Simple app to recreate figure 6 from Wapenaar et al., 2010A
"""

import numpy as np

from viz import plot_map, plot_gather

from sources import get_gather


def generate_event_locations(
        station_locations,
        angle_1=-90,
        angle_2=270,
        radius_1=2_000,
        radius_2=3_000,
        num_events=500,
):
    """Generate random event locations from centroid of stations"""
    centroid = np.mean(station_locations, axis=0)
    angles = np.sort(np.random.uniform(angle_1, angle_2, size=num_events))
    radii = np.random.uniform(radius_1, radius_2, size=num_events)

    x_values = np.sin(np.deg2rad(angles)) * radii
    y_values = np.cos(np.deg2rad(angles)) * radii

    return centroid + np.stack([x_values, y_values], axis=-1), angles


def create_gathers(
        station_array,
        event_array,
        source_type='ricker',
        velocity=2_000,
        duration=2.0,
        dt=0.001,
        **kwargs,
):
    """Create gathers based on specified source time function."""
    out = []
    for station in station_array:
        gather = get_gather(
            station,
            event_array,
            duration=duration,
            dt=dt,
            velocity=velocity,
            source_type=source_type,
        )
        out.append(gather)
    return out


def correlate(gather_1, gather_2):
    """Correlate gathers together."""
    assert gather_1.shape == gather_2.shape
    shape = gather_1.shape

    # first pad each array with 0s
    padded_1 = np.zeros((shape[0] * 2, shape[1]), dtype=np.float64)
    padded_1[:shape[0], :] = gather_1.values
    padded_2 = np.zeros((shape[0] * 2, shape[1]), dtype=np.float64)
    padded_2[:shape[0], :] = gather_2.values

    # the transform to fourier domain
    fft1 = np.fft.rfft(gather_1, axis=0)
    fft2 = np.fft.rfft(gather_2, axis=0)

    # perform correlation in freq domain
    out_fft = fft1 * np.conj(fft2)

    out_time = np.fft.irfft(out_fft, axis=0)
    out = np.fft.fftshift(out_time, axes=0)

    breakpoint()


if __name__ == "__main__":
    velocity = 2_000  # m /s
    station_array = np.array([[-600, 0], [600, 0]])  # in m
    event_array, phi = generate_event_locations(station_array)

    gathers = create_gathers(station_array, event_array)

    correlation = correlate(gathers[0], gathers[1])



    for gather in gathers:
        plot_gather(gather, phi)

    plot_map(station_array, event_array)
    breakpoint()









