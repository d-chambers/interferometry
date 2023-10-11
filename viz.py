import numpy as np
from matplotlib import pyplot as plt


def plot_map(station_locations, event_locations, ax=None):
    """Plot event and station locations."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(event_locations[:, 0], event_locations[:, 1], '.', color="#1f77b4")
    ax.plot(station_locations[:, 0], station_locations[:, 1], '^', color="#ff7f0e")
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.show()


def plot_gather(gather, phi, ax=None, exaggeration=3):
    """Plot the gather. """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

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
    new_x_ticks = np.linspace(start=-90, stop=270, num=5)
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

    # remove white space around margins
    ax.margins(x=0)
    ax.margins(y=0)

    return ax
