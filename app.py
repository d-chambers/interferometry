"""
Simple app to recreate figure 6 from Wapenaar et al., 2010A
"""
import numpy as np

from core import generate_event_df, create_gathers, correlate
from viz import plot_map, plot_gather, create_master_layout, plot_corr_trace

import streamlit as st

st.set_page_config(page_title="2D Interferometry", page_icon=":eyeglasses:")




if __name__ == "__main__":
    st.sidebar.markdown("## Select Parameters")
    velocity = st.sidebar.text_input("velocity (m/s)", value=2_000)

    station_array = np.array([[-600, 0], [600, 0]])  # in m

    fig, ax_dict = create_master_layout(figsize=(24, 12))
    event_df = generate_event_df(station_array)

    gathers = create_gathers(station_array, event_df, velocity=float(velocity))
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
    plot_corr_trace(corr_stack, ax=ax_dict["ax_stacked_correlation"])
    st.pyplot(fig=fig)









