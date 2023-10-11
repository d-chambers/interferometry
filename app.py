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


if __name__ == "__main__":
    velocity = 2_000  # m /s
    station_array = np.array([[-600, 0], [600, 0]])  # in m
    event_array, phi = generate_event_locations(station_array)

    gathers = create_gathers(station_array, event_array)
    for gather in gathers:


        plot_gather(gather, phi)

    plot_map(station_array, event_array)
    breakpoint()









