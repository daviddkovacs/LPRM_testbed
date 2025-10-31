import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import os


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


def scatter_plot(ref,
                 test,
                 xlabel = "",
                 ylabel = "",
                 xmin_val=0,
                 xmax_val= 0.08,
                 ymin_val = 0,
                 ymax_val = 0.08,
                 savedir = None,
                 showfig = True):

    stats_dict = statistics(ref,test)

    stats_text = (f"R: {stats_dict['r']}\nRMSE: {stats_dict['rmse']}\n"
                  f"Bias: {stats_dict['bias']}\n"
                  f"Precision: {stats_dict['precision']}\n"
                  f"N: {stats_dict['N']}\n")

    mask = np.isfinite(ref) & np.isfinite(test)
    ref = ref[mask]
    test = test[mask]

    xy = np.vstack([ref, test])
    z = gaussian_kde(xy)(xy)

    plt.figure(figsize=(6, 6))

    plt.scatter(ref, test, c=z, s=20, cmap='viridis', )

    plt.plot([xmin_val, xmax_val], [ymin_val, ymax_val], 'k-', lw=1, )

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left', fontsize=14)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.xlim([xmin_val, xmax_val])
    plt.ylim([ymin_val, ymax_val])
    plt.tight_layout()
    if savedir:
        plt.savefig(os.path.join(savedir,"scatter.png"))
    if showfig:
        plt.show()


def longitude_plot(ref_x,
                   ref_y,
                   test_x,
                   test_y,
                   test2_x,
                   test2_y,
                   air_obj,
                   sat_obj,
                   bio_obj,
                   savedir = None,
                   show_fig = True
                   ):

    # Satellite variables
    sat_freq = sat_obj.sat_freq
    sat_sensor = sat_obj.sat_sensor
    target_res = sat_obj.target_res

    # Airborne variables
    flight_direction = air_obj.flight_direction
    air_freq = air_obj.air_freq
    scan_direction = air_obj.scan_direction

    # Bio variables
    bio_var = bio_obj.variable

    # Common variables
    date = air_obj.air_freq
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
             label=f"ERA5 {bio_var}",
             color="tab:brown",
             linestyle='--',
             markersize=2)
    ax2.set_ylabel(f"ERA5 {bio_var}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title(f"{date} {flight_direction} {scan_direction}\n"
              f"R: {stats_dict['r']}\n"
              f"RMSE: {stats_dict['rmse']}\n"
              f"Bias: {stats_dict['bias']}\n"
              )

    plt.grid(False)
    plt.tight_layout()
    if savedir:
        plt.savefig(os.path.join(savedir,rf"{date}_{flight_direction}_{scan_direction}_{air_freq}_long.png"))
    if show_fig:
        plt.show()

