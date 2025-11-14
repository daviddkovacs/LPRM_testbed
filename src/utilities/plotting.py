import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import os
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap


def statistics(ref,test):

    r = pd.Series(ref).corr(pd.Series(test))
    rmse = np.sqrt(np.mean((test - ref) ** 2))
    bias = np.mean(ref) - np.mean(test)
    precision = np.round(np.sqrt(np.mean(
        (test - ref - np.mean(
            test - ref)) ** 2)),
        2)

    stats_dict = {"r": np.round(r, 2),
                  "rmse": np.round(rmse, 3),
                  "bias": np.round(bias, 3),
                  "precision": np.round(precision, 3),
                  "N": len(ref)}

    return stats_dict


def create_scatter_plot(ref,
                        test,
                        test_colour=None,
                        xlabel = None,
                        ylabel = None,
                        cbar_label="Density",
                        xlim=(None, None),
                        ylim=(None, None),
                        cbar_scale=(None, None),
                        stat_text = True,
                        ):


    mask = np.isfinite(ref) & np.isfinite(test)
    if test_colour is not None:
        mask &= np.isfinite(test_colour)

    ref = ref[mask]
    test = test[mask]
    test_colour = test_colour[mask] if test_colour is not None else None

    if test_colour is None:
        xy = np.vstack([ref, test])
        z = gaussian_kde(xy)(xy)
    else:
        z = test_colour

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(ref, test, c=z, s=20, cmap='turbo', vmin= cbar_scale[0], vmax = cbar_scale[1])

    if stat_text:
        plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], 'k-', lw=1)

        stats_dict = statistics(ref, test)
        stats_text = (f"R: {stats_dict['r']}\nRMSE: {stats_dict['rmse']}\n"
                      f"Bias: {stats_dict['bias']}\n"
                      f"Precision: {stats_dict['precision']}\n"
                      f"N: {stats_dict['N']}\n")
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', horizontalalignment='left', fontsize=14)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.xlim([xlim[0], xlim[1]])
    plt.ylim([ylim[0], ylim[1]])
    plt.colorbar(sc, label=cbar_label)
    plt.tight_layout()
    plt.show()


def scatter_density(ref,
                    test,
                    test_colour=None,
                    xlabel=None,
                    ylabel=None,
                    cbar_label="Density",
                    cbar_type=None,
                    xlim=(None, None),
                    ylim=(None, None),
                    cbar_scale=(None, None),
                    dpi = 30
                    ):

    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    if cbar_type is None:
        cbar_type = white_viridis

    # fig = plt.figure(dpi=300, figsize=(6, 4))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    # fig.set_size_inches(4, 3, forward=True)

    density = ax.scatter_density(ref,
                                 test,
                                 c=test_colour,
                                 cmap=cbar_type,
                                 dpi=30,
                                 vmin=cbar_scale[0],
                                 vmax=cbar_scale[1],
                                 )

    fig.colorbar(density, label= cbar_label)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.canvas.draw_idle()
    plt.pause(0.001)
    return fig, ax, density


def create_longitude_plot(ref_x,
                   ref_y,
                   test_x,
                   test_y,
                   test2_x,
                   test2_y,
                   show_fig = True,
                          **kwargs
                   ):


    # Satellite variables
    sat_freq = kwargs.get("sat_freq")
    sat_sensor = kwargs.get("sat_sensor")
    target_res = kwargs.get("target_res")

    # Airborne variables
    flight_direction = kwargs.get("flight_direction")
    air_freq = kwargs.get("air_freq")
    scan_direction = kwargs.get("scan_direction")

    # Bio variables
    bio_var = kwargs.get("bio_var")

    # Common variables
    date = kwargs.get("date")
    stats_dict = statistics(ref_y,test_y)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(ref_x, ref_y,
             label=f"AMPR {air_freq} GHz",
             color="tab:blue")
    ax1.plot(test_x, test_y,
             label=f"{sat_sensor.upper()} {target_res}km {sat_freq} GHz",
             color="tab:orange",
             marker='x',
             linestyle='',
             markersize=6)

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("MPDI")
    ax1.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(test2_x, test2_y,
             label=f"{bio_var}",
             color="tab:brown",
             linestyle='--',
             markersize=2)
    ax2.set_ylabel(f"{bio_var}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title(f"{date} {flight_direction} {scan_direction}\n"
              )

    plt.grid(False)
    plt.tight_layout()
    if show_fig:
        plt.show()


def plot_maps(df, cbar_lut):

    cordinates = [df["LAT"].values, df["LON"].values]
    mi_array = zip(*cordinates)
    df.index = pd.MultiIndex.from_tuples(mi_array, names=["LAT", "LON"])

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    for ax, var in zip(axes, cbar_lut.keys()):
        data = df.to_xarray()[var]
        im = ax.imshow(np.flipud(data), vmin=cbar_lut[var][0], vmax=cbar_lut[var][1])
        ax.set_title(var)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


