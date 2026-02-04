import numpy as np
import pandas as pd
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

    ax_lst = plt.subplot(3, 2, 1)
    left_da.plot.pcolormesh(
        x=left_params["x"],
        y=left_params["y"],
        ax=ax_lst,
        cmap=left_params["cmap"],
        cbar_kwargs=left_params["cbar_kwargs"],
        vmin=left_params["vmin"],
        add_colorbar=True
    )
    ax_lst.set_title(left_params["title"])

    ax_ndvi = plt.subplot(3, 2, 2)
    right_da.plot.pcolormesh(
        x=right_params["x"],
        y=right_params["y"],
        ax=ax_ndvi,
        cmap=right_params["cmap"],
        cbar_kwargs=right_params["cbar_kwargs"],
        vmin=right_params.get("vmin"),
        vmax=right_params.get("vmax"),
        add_colorbar=True
    )
    ax_ndvi.set_title(right_params["title"])
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
    plt.title(f"AMSR2 LST\n{ds.time.dt.strftime("%Y-%m-%d").item()}")
    plt.show()



def temps_plot(df, plot_mpdi = False):
    """
    combined plot with Ka-band microwave temperatures, Vegetation, and Soil temps.
    """
    x = np.arange(len(df))

    plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(2, 2, (1, 2))
    ax1.plot(x, df["veg_mean"], label='Vegetation Mean', color='forestgreen', linewidth=2)
    ax1.fill_between(x,
                     np.array(df["veg_mean"]) - np.array(df["veg_std"]),
                     np.array(df["veg_mean"]) + np.array(df["veg_std"]),
                     color='forestgreen', alpha=0.2)

    ax1.plot(x, df["soil_mean"], label='Soil Mean', color='saddlebrown', linewidth=2)
    ax1.fill_between(x,
                     np.array(df["soil_mean"]) - np.array(df["soil_std"]),
                     np.array(df["soil_mean"]) + np.array(df["soil_std"]),
                     color='saddlebrown', alpha=0.2)
    ax1.plot(x, df["tsurf_ka"], label='Ka TSURF', color='red', linewidth=2)
    if "mpdi" in df.columns and plot_mpdi:
        ax2 = ax1.twinx()
        ax2.plot(x, df["mpdi"], label='MPDI', color='blue', linewidth=2, alpha = 0.5)
        ax2.set_ylabel('MPDI', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

    ax1.set_ylabel(r'Surface Temperature $[K]$')
    ax1.set_xlabel(r'# of AMSR2 pixels')
    ax1.set_title('Sub-pixel LST Statistics per AMSR2 pixel')
    ax1.legend(loc='upper left', frameon=True)

    ax2 = plt.subplot(2, 2, 3)
    ax2.scatter(df["tsurf_ka"], df["veg_mean"],)
    ax2.set_xlim([270, 320])
    ax2.set_ylim([270, 320])
    ax2.set_xlabel(r'AMSR2 Ka $T$ $[K]$')
    ax2.set_ylabel(r'Veg. $T$ $[K]$')
    ax2.set_title('Temperatures: Ka - Vegetation')

    ax3 = plt.subplot(2, 2, 4)
    ax3.scatter(df["tsurf_ka"], df["soil_mean"],)
    ax3.set_xlim([270, 320])
    ax3.set_ylim([270, 320])
    ax3.set_xlabel(r'AMSR2 Ka $T$ $[K]$')
    ax3.set_ylabel(r'Soil $T$ $[K]$')
    ax3.set_title('Temperatures: Ka - Soil')

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np


def combined_dashboard(left_da, right_da, left_params, right_params, df, bbox=None, plot_mpdi=False,
                       plot_scatter=False):
    """
    Combines Sentinel-3 spatial plots (LST/NDVI) and AMSR2/LST statistical plots into one figure.
    """
    if 'time' in left_da.dims:
        if left_da.sizes['time'] > 1:
            left_da = left_da.isel(time=0)
        else:
            left_da = left_da.squeeze('time')

    if 'time' in right_da.dims:
        if right_da.sizes['time'] > 1:
            right_da = right_da.isel(time=0)
        else:
            right_da = right_da.squeeze('time')

    # try:
    obs_date = pd.to_datetime(left_da.time.values).strftime('%Y-%m-%d')
    # except:
    #     obs_date = "Selected Date"

    fig = plt.figure(figsize=(14, 10))

    ax_lst = plt.subplot(2, 2, 1)
    left_da.plot.pcolormesh(
        x=left_params["x"],
        y=left_params["y"],
        ax=ax_lst,
        cmap=left_params["cmap"],
        cbar_kwargs=left_params["cbar_kwargs"],
        vmin=left_params["vmin"],
        add_colorbar=True
    )
    ax_lst.set_title(left_params["title"])

    ax_ndvi = plt.subplot(2, 2, 2)
    right_da.plot.pcolormesh(
        x=right_params["x"],
        y=right_params["y"],
        ax=ax_ndvi,
        cmap=right_params["cmap"],
        cbar_kwargs=right_params["cbar_kwargs"],
        vmin=right_params.get("vmin"),
        vmax=right_params.get("vmax"),
        add_colorbar=True
    )
    ax_ndvi.set_title(right_params["title"])

    # Draw Bounding Box if provided
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        for ax in [ax_lst, ax_ndvi]:
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='red',
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)

    ax1 = plt.subplot(2, 2, (3, 4))
    x_idx = np.arange(len(df))

    ax1.plot(x_idx, df["veg_mean"], label='Vegetation Mean', color='forestgreen', linewidth=2)
    ax1.fill_between(x_idx,
                     df["veg_mean"] - df["veg_std"],
                     df["veg_mean"] + df["veg_std"],
                     color='forestgreen', alpha=0.2)

    # Soil Stats
    ax1.plot(x_idx, df["soil_mean"], label='Soil Mean', color='saddlebrown', linewidth=2)
    ax1.fill_between(x_idx,
                     df["soil_mean"] - df["soil_std"],
                     df["soil_mean"] + df["soil_std"],
                     color='saddlebrown', alpha=0.2)

    # AMSR2 Reference
    ax1.plot(x_idx, df["tsurf_ka"], label='Ka TSURF', color='red', linewidth=2,)

    ax1.set_ylabel(r'Temperature $[K]$')
    ax1.set_xlabel('AMSR2 Pixel Index')
    ax1.set_title('Sub-pixel LST Statistics per AMSR2 pixel')
    ax1.legend(loc='upper left', frameon=True)

    # Secondary Axis for MPDI
    if "mpdi" in df.columns and plot_mpdi:
        ax_mpdi = ax1.twinx()
        ax_mpdi.plot(x_idx, df["mpdi"], label='MPDI', color='blue', linewidth=1.5, alpha=0.5)
        ax_mpdi.set_ylabel('MPDI', color='blue')
        ax_mpdi.tick_params(axis='y', labelcolor='blue')


    plt.suptitle(f"Sentinel-3 SLSTR and AMSR2 Analysis | {obs_date}", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()