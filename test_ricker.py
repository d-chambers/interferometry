"""
Offline (no streamlit) run.
"""
import matplotlib.pyplot as plt
import numpy as np

from core import generate_event_df, create_gathers, correlate
from viz import plot_map, plot_gather, create_master_layout, plot_corr_trace


if __name__ == "__main__":
    velocity = 2_000  # m /s
    station_array = np.array([[-600, 0], [600, 0]])  # in m

    fig, ax_dict = create_master_layout()
    event_df = generate_event_df(station_array)

    gathers = create_gathers(station_array, event_df, velocity=velocity, source_type="ricker")
    correlation = correlate(gathers[0], gathers[1])
    correlation = correlation.loc[slice(-1, 1)]
    corr_stack = correlation.sum(axis=1)

    for num, gather in enumerate(gathers):
        ax = ax_dict[f"ax_gather_{num + 1}"]
        plot_gather(gather, title=f"Gather {num + 1}", ax=ax)

    plot_gather(
        correlation,
        title="Correlations",
        ax=ax_dict["ax_correlations"],
    )

    plot_map(station_array, event_df, ax=ax_dict["ax_map"])
    plot_corr_trace(corr_stack, station_array, ax=ax_dict["ax_stacked_correlation"])
    plt.show()
