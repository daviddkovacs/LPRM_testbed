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
    plt.title(f"AMSR2 LST\n{ds.time.dt.strftime("%Y-%m-%d").item()}")
    plt.show()



def temps_plot(df):
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
