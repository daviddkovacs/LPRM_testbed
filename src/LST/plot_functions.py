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
                   "vmin": 273,
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


def combined_dashboard(LST_L1,
                       NDVI_L1,
                       LST_params,
                       NDVI_params,
                       df_S3_pixels_in_AMSR2,
                       bbox=None,
                       plot_mpdi=False,
                       plot_kuka=False,
                       mpdi_band =None):
    """
    Combines Sentinel-3 spatial plots (LST/NDVI) and AMSR2/LST statistical plots into one figure.
    """
    if 'time' in LST_L1.dims:
        if LST_L1.sizes['time'] > 1:
            LST_L1 = LST_L1.isel(time=0)
        else:
            LST_L1 = LST_L1.squeeze('time')

    if 'time' in NDVI_L1.dims:
        if NDVI_L1.sizes['time'] > 1:
            NDVI_L1 = NDVI_L1.isel(time=0)
        else:
            NDVI_L1 = NDVI_L1.squeeze('time')

    obs_date = pd.to_datetime(LST_L1.time.values).strftime('%Y-%m-%d')

    fig = plt.figure(figsize=(14, 10))
    ax_lst = plt.subplot(2, 2, 1)
    LST_L1.plot.pcolormesh(
        x=LST_params["x"],
        y=LST_params["y"],
        ax=ax_lst,
        cmap=LST_params["cmap"],
        cbar_kwargs=LST_params["cbar_kwargs"],
        vmin=LST_params["vmin"],
        add_colorbar=True
    )
    ax_lst.set_title(LST_params["title"])

    ax_ndvi = plt.subplot(2, 2, 2)
    NDVI_L1.plot.pcolormesh(
        x=NDVI_params["x"],
        y=NDVI_params["y"],
        ax=ax_ndvi,
        cmap=NDVI_params["cmap"],
        cbar_kwargs=NDVI_params["cbar_kwargs"],
        vmin=NDVI_params.get("vmin"),
        vmax=NDVI_params.get("vmax"),
        add_colorbar=True
    )
    ax_ndvi.set_title(NDVI_params["title"])

    if bbox:
        xmin, ymin, xmax, ymax = bbox
        for ax in [ax_lst, ax_ndvi]:
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='red',
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)

    ax1 = plt.subplot(2, 2, (3, 4))
    x_idx = np.arange(len(df_S3_pixels_in_AMSR2))

    # Vegetation Stats
    ax1.plot(x_idx, df_S3_pixels_in_AMSR2["veg_temp"],
             label='Vegetation Mean', color='forestgreen', linewidth=2)
    ax1.fill_between(x_idx,
                     df_S3_pixels_in_AMSR2["veg_temp"] - df_S3_pixels_in_AMSR2["veg_std"],
                     df_S3_pixels_in_AMSR2["veg_temp"] + df_S3_pixels_in_AMSR2["veg_std"],
                     color='forestgreen', alpha=0.2)

    # Soil Stats
    ax1.plot(x_idx, df_S3_pixels_in_AMSR2["soil_temp"],
             label='Soil Mean', color='saddlebrown', linewidth=2)
    ax1.fill_between(x_idx,
                     df_S3_pixels_in_AMSR2["soil_temp"] - df_S3_pixels_in_AMSR2["soil_std"],
                     df_S3_pixels_in_AMSR2["soil_temp"] + df_S3_pixels_in_AMSR2["soil_std"],
                     color='saddlebrown', alpha=0.2)

    # AMSR2 Ka Holmes TSURF
    ax1.plot(x_idx, df_S3_pixels_in_AMSR2["tsurf_ka"], label='Ka TSURF', color='red', linewidth=2, )

    # AMSR2 Adjusted TSURF
    ax1.plot(x_idx, df_S3_pixels_in_AMSR2["tsurf_adj"],
              label='Adj. TSURF', color='magenta', linewidth=1.5)

    ax1.set_ylabel(r'Temperature $[K]$')
    ax1.set_xlabel('AMSR2 Pixel Number in ROI')
    ax1.set_title('Sub-pixel LST per AMSR2 pixel')
    ax1.legend(loc='upper left', frameon=True)

    # Secondary Axis for MPDI
    if plot_mpdi:
        ax_mpdi = ax1.twinx()
        ax_mpdi.plot(x_idx, df_S3_pixels_in_AMSR2["mpdi"],
                     label='MPDI', color='blue', linewidth=1.5)
        ax_mpdi.set_ylabel(f'MPDI {mpdi_band}', color='blue')
        ax_mpdi.tick_params(axis='y', labelcolor='blue')

    if plot_kuka:
        ax_kuka = ax1.twinx()
        # Offset the right spine of the second twin axis so it's not on top of MPDI
        ax_kuka.spines["right"].set_position(("axes", 1.1))

        ax_kuka.plot(x_idx, df_S3_pixels_in_AMSR2["kuka"],
                     label='KuKa', color='darkorange', linewidth=1.5)
        ax_kuka.set_ylabel('KuKa Index', color='darkorange')
        ax_kuka.tick_params(axis='y', labelcolor='darkorange')

        plt.subplots_adjust(right=0.85)

    plt.suptitle(f"Sentinel-3 SLSTR and AMSR2 comparison | {obs_date}", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_scatter(df, x_col="soil_temp", y_col="tsurf_adj"):

    r = df[x_col].corr(df[y_col])
    error = df[y_col] - df[x_col]
    bias = error.mean()
    rmse = np.sqrt((error ** 2).mean())

    plt.figure(figsize=(6, 5))
    plt.scatter(df[x_col], df[y_col], alpha=0.8, s=15)

    lims = [df[[x_col, y_col]].min().min(), df[[x_col, y_col]].max().max()]
    plt.plot(lims, lims, 'r--', label="1:1 Line")

    stats_str = f"r: {r:.3f}\nRMSE: {rmse:.3f}\nBias: {bias:.3f}"
    plt.text(0.05, 0.85, stats_str, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white')
             )
    plt.title(f"x: {x_col}   y: {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()