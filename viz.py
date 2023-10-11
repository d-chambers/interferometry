import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec


def create_master_layout(figsize=(12, 6)):
    """Creates a layout for the plots."""
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(2, 4, figure=fig)
    out = dict(
        ax_map=fig.add_subplot(gs[0:2, 0:2]),
        # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
        ax_gather_1=fig.add_subplot(gs[0, 2]),
        ax_gather_2=fig.add_subplot(gs[1, 2]),
        ax_correlations=fig.add_subplot(gs[0, 3]),
        ax_stacked_correlation=fig.add_subplot(gs[1, 3]),
    )
    return fig, out


def plot_map(station_locations, event_df, ax=None):
    """Plot event and station locations."""
    event_locations = event_df[['x', 'y']].values
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(event_locations[:, 0], event_locations[:, 1], '.', color="#1f77b4")
    ax.plot(station_locations[:, 0], station_locations[:, 1], '^', color="#ff7f0e")
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for num, sta in enumerate(station_locations):
        text = f"Sta {num + 1}"
        ax.annotate(text, sta, ha='center', va="bottom")

    return ax


def plot_gather(gather, ax=None, exaggeration=3, title=None):
    """Plot the gather. """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    phi = gather.columns.values

    # normalize data to plot with offsets.
    time = gather.index.values
    data = gather.values
    excursion = data.max(axis=0) - data.min(axis=0)
    data_normed = (exaggeration * data) / excursion[None, :]
    offsets = np.arange(data.shape[1])
    data_offset = data_normed + offsets[None, :]

    # ax.plot(gather.index.values, data_offset, color='grey', alpha=0.1)
    ax.plot(data_offset, time, color='grey', alpha=0.1)
    ax.invert_yaxis()
    ax.set_ylabel("time")
    ax.set_xlabel("$\phi_s$")

    # set x labels to phi
    x_lims = ax.get_xlim()
    replace_x_ticks = np.linspace(x_lims[0], x_lims[1], num=5)
    new_x_ticks = np.linspace(
        start=np.round(np.abs(phi.min())) * np.sign(phi.min()),
        stop=np.round(phi.max()),
        num=5
    )
    ax.set_xticks(replace_x_ticks, new_x_ticks.astype(np.int64))

    # fill in values above mean
    for i in range(len(offsets)):
        ax.fill_betweenx(
            time,
            offsets[i],
            data_offset[:, i],
            where=(data_offset[:, i] > offsets[i]),
            color='black',
            alpha=0.6,
        )

    # remove white space around margins and set title
    ax.margins(x=0)
    ax.margins(y=0)
    if title is not None:
        ax.set_title(title)

    return ax


def plot_corr_trace(corr_stack, ax=None):
    """Plot the entire correlation stack."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(corr_stack.index.values, corr_stack.values)
    ax.set_title("Correlation Stack")
    return ax