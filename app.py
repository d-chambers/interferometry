"""
Simple app to recreate figure 6 from Wapenaar et al., 2010A
"""
import numpy as np
import matplotlib.pyplot as plt

from core import generate_event_df, create_gathers, correlate
from sources import SUPPORTED_SOURCE_TYPES
from viz import plot_map, plot_gather, create_master_layout, plot_corr_trace

import streamlit as st

st.set_page_config(page_title="2D Interferometry", page_icon=":eyeglasses:")

# Configure plots
st.title("2D Interferometry")


# default params used in condition inputs
frequency = 30
freq_min = None
freq_max = None


### setup side-bar with input params.
st.sidebar.markdown("## Geometry Parameters")
velocity = float(st.sidebar.text_input("velocity (m/s)", value=2_000))
number_of_sources = int(st.sidebar.text_input("source count", value=500))
angle_range = st.sidebar.slider("Angle Range", -90, 270, value=(-90, 270))
radius_range = st.sidebar.slider("Radius", 0, 8_000, value=(2_000, 3_000))

# setup sources
st.sidebar.markdown("## Source Parameters")
source_type = st.sidebar.selectbox(
    "Source Type", tuple(SUPPORTED_SOURCE_TYPES)
)
if source_type == "ricker":
    frequency = float(st.sidebar.text_input("Frequency (Hz)", 30))
elif source_type == "random":
    freq_min = st.sidebar.text_input("Minimum Frequency (Hz)", None)
    freq_max = st.sidebar.text_input("Maximum Frequency (Hz)", None)


### Create and plot map
station_array = np.array([[-600, 0], [600, 0]])  # in m
event_df = generate_event_df(
    station_array,
    num_events=int(number_of_sources),
    angle_1=float(angle_range[0]),
    angle_2=float(angle_range[1]),
    radius_1=float(radius_range[0]),
    radius_2=float(radius_range[1]),
)
ax = plot_map(station_array, event_df, lims=radius_range)
st.pyplot(fig=ax.figure, clear_figure=True)


### Create and plot gathers
fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), sharey=True)
gathers = create_gathers(
    station_array,
    event_df,
    velocity=float(velocity),
    source_type=source_type,
    frequency=frequency,
    freq_min=freq_min,
    freq_max=freq_max,
)
for num, gather in enumerate(gathers):
    ax = axes[num]
    plt.subplots_adjust(hspace=0.01, wspace=0.03)
    plot_gather(gather, title=f"Gather {num + 1}", ax=ax)
ax.set_ylabel("")
st.pyplot(fig=fig, clear_figure=True)


### plot correlation gather
correlation_df = correlate(gathers[0], gathers[1])
ax = plot_gather(correlation_df)
ax.set_title("Correlation Gather")
st.pyplot(fig=ax.figure, clear_figure=True)


### stack gathers and plot correlation stack
corr_ser = correlation_df.sum(axis=1)
ax = plot_corr_trace(corr_ser, station_array, velocity)
st.pyplot(fig=ax.figure, clear_figure=True)

