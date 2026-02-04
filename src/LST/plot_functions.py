import glob
import re
from functools import partial
import numpy as np
import xarray as xr
import os
from xarray import apply_ufunc
from config.paths import SLSTR_path
import pandas as pd
from datetime import datetime
from LST.SLSTR_utils import filternan, subset_statistics
import matplotlib.patches as patches
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

LST_plot_params = {"x": "lon",
                   "y": "lat",
                   "cmap": "coolwarm",
                   "cbar_kwargs": {'label': 'LST [K]'},
                   "vmin": 290,
                   "title": "LST"
                   }
NDVI_plot_params = {
    "x": "lon",
    "y": "lat",
    "cmap": "YlGn",
    "cbar_kwargs": {'label': "NDVI [-]"},
    "vmin": 0,
    "vmax": 0.6,
    "title": "NDVI"
}
AMSR2_plot_params = {
    "cmap": "coolwarm",
    "cbar_kwargs": {'label': 'LST [K]'},
    "vmin": 290,
    "vmax": 320,
}

def plot_lst(left_da,
             right_da,
             left_params,
             right_params,
             bbox = None
             ):

    pd.to_datetime(left_da.time.values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    obs_date = pd.to_datetime(left_da.time.values)

    left_da.plot(
        x=left_params["x"],
        y=left_params["y"],
        ax=ax1,
        cmap=left_params["cmap"],
        cbar_kwargs=left_params["cbar_kwargs"],
        vmin=left_params["vmin"]
    )
    ax1.set_title(left_params["title"])

    right_da.plot(
        x=right_params["x"],
        y=right_params["y"],
        ax=ax2,
        cmap=right_params["cmap"],
        cbar_kwargs=right_params["cbar_kwargs"],
        vmin=right_params["vmin"],
        vmax=right_params["vmax"]
    )
    ax2.set_title(right_params["title"])
    plt.suptitle(f"Sentinel-3 SLSTR\n{obs_date}")

    if bbox:
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        for ax in [ax1, ax2]:
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=2, edgecolor='red',
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)
    plt.show()


def plot_amsr2(ds,
               plot_params):

    plt.figure()
    ds.plot(
        cmap= plot_params["cmap"],
        cbar_kwargs=plot_params["cbar_kwargs"],
        vmin=plot_params["vmin"],
        vmax=plot_params["vmax"]
    )
    plt.title(f"AMSR2 LST {ds.time}")
    plt.show()


def boxplot_soil_veg(soil, veg, ndvi_thres=0.3, bins =200):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    soil_filtered, _ = subset_statistics(soil)
    veg_filtered, _ = subset_statistics(veg)

    ax1.hist(soil_filtered,
             bins=bins,
             alpha=0.8,
             label=f"$T_{{soil}}$ (NDVI < {ndvi_thres})",
             color="brown")
    ax1.hist(veg_filtered,
             bins=bins,
             alpha=0.7,
             label=f"$T_{{vegetation}}$ (NDVI > {ndvi_thres})",
             color="green")
    ax1.set_xlabel("$T$ [K]")
    ax1.set_ylabel("frequency")
    ax1.set_title("Temp Distribution")
    ax1.legend(loc="upper left")

    data_to_plot = [filternan(soil), filternan(veg)]
    bp = ax2.boxplot(data_to_plot,
                     patch_artist=True,
                     showfliers=False,
                     tick_labels=[f"Soil", f"Veg"])

    colors = ["brown", "green"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel("$T$ [K]")
    ax2.set_title("Soil/Veg. Temp Boxplot")

    plt.tight_layout()
    plt.show()


def temps_plot(df):
    """
    create a plot with Ka-band microwave temps, Vegetation (higher NDVI) and soil (lower NDVI) temperatures.
    This allows to compare how these differ.
    """
    x = np.arange(len(df))

    plt.figure()
    plt.plot(x, df["tsurf_ka"], label='Ka TSURF', color='red', linewidth=2)

    plt.plot(x, df["veg_mean"], label='Vegetation Mean', color='forestgreen', linewidth=2)
    plt.fill_between(x,
                     np.array(df["veg_mean"]) - np.array(df["veg_std"]),
                     np.array(df["veg_mean"]) + np.array(df["veg_std"]),
                     color='forestgreen', alpha=0.2,)

    plt.plot(x, df["soil_mean"], label='Soil Mean', color='saddlebrown', linewidth=2)
    plt.fill_between(x,
                     np.array(df["soil_mean"]) - np.array(df["soil_std"]),
                     np.array(df["soil_mean"]) + np.array(df["soil_std"]),
                     color='saddlebrown', alpha=0.2, )

    plt.ylabel('Surface Temperature [K]')
    plt.title('Sub-pixel LST Statistics per  AMSR2 pixel')
    plt.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.show()

