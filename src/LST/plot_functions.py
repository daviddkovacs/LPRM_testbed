import numpy as np
import pandas as pd
import matplotlib.patches as patches
import matplotlib
import matplotlib.pyplot as plt
from comparison_utils import subset_statistics
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


def usual_stats(x,y):
    r =x.corr(y)
    bias = (y - x).mean()
    rmse = np.sqrt(((y - x) ** 2).mean())
    return {"r" : r , "bias" : bias , "rmse" : rmse}


def plot_hexbin(df, x_col, y_col, title=None, gridsize=100, cmap='inferno'):
    x = df[x_col]
    y = df[y_col]
    stats = usual_stats(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))

    hb = ax.hexbin(x, y, bins="log", gridsize=gridsize, cmap=cmap, mincnt=1)

    lims = [273, 325]
    ax.plot(lims, lims, 'k--', alpha=0.8, linewidth=1, zorder=10)  # 'k--' is black dashed

    textstr = '\n'.join((
        f'$R = {stats["r"]:.2f}$',
        f'$Bias = {stats["bias"]:.2f}$ K',
        f'$RMSE = {stats["rmse"]:.2f}$ K'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title if title else f'{x_col} vs {y_col}')


    plt.show()


def combined_validation_dashboard(LST_L1,
                                  NDVI_L1,
                                  LST_params,
                                  NDVI_params,
                                  df_S3_pixels_in_AMSR2,
                                  bbox=None,
                                  plot_mpdi=False,
                                  plot_tsurf_adjust = False,
                                  plot_kuka=False,
                                  mpdi_band=None,
                                  scatter_x = None,
                                  ):
    """
    Combines spatial plots, pixel-wise time series, and 1:1 scatter validation.
    """
    LST_L1 = LST_L1.isel(time=0) if 'time' in LST_L1.dims and LST_L1.sizes['time'] > 1 else LST_L1.squeeze()
    NDVI_L1 = NDVI_L1.isel(time=0) if 'time' in NDVI_L1.dims and NDVI_L1.sizes['time'] > 1 else NDVI_L1.squeeze()
    obs_date = pd.to_datetime(LST_L1.time.values).strftime('%Y-%m-%d')

    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(3, 2)

    ax_lst = fig.add_subplot(gs[0, 0])
    LST_L1.plot.pcolormesh(x=LST_params["x"], y=LST_params["y"], ax=ax_lst, cmap=LST_params["cmap"],
                           vmin=LST_params["vmin"], add_colorbar=True, cbar_kwargs=LST_params["cbar_kwargs"])
    ax_lst.set_title(LST_params["title"])

    ax_ndvi = fig.add_subplot(gs[0, 1])
    NDVI_L1.plot.pcolormesh(x=NDVI_params["x"], y=NDVI_params["y"], ax=ax_ndvi, cmap=NDVI_params["cmap"],
                            vmin=NDVI_params.get("vmin"), vmax=NDVI_params.get("vmax"), add_colorbar=True)
    ax_ndvi.set_title(NDVI_params["title"])

    if bbox:
        xmin, ymin, xmax, ymax = bbox
        for ax in [ax_lst, ax_ndvi]:
            ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))

    _, tka_s = subset_statistics(df_S3_pixels_in_AMSR2["tsurf_ka"])
    _, tadj_s = subset_statistics(df_S3_pixels_in_AMSR2["tsurf_adj"])
    _, soil_s = subset_statistics(df_S3_pixels_in_AMSR2["soil_temp"])
    _, veg_s = subset_statistics(df_S3_pixels_in_AMSR2["veg_temp"])

    ax1 = fig.add_subplot(gs[1, :])  # Span both columns
    x_idx = np.arange(len(df_S3_pixels_in_AMSR2))

    #Veg. stats
    ax1.plot(x_idx, df_S3_pixels_in_AMSR2["veg_temp"],
             label=f'Veg T ({veg_s["mean"]:.1f}±{veg_s["std"]:.1f}K)',
             color='forestgreen', linewidth=2)
    ax1.fill_between(x_idx,
                     df_S3_pixels_in_AMSR2["veg_temp"] - df_S3_pixels_in_AMSR2["veg_std"],
                     df_S3_pixels_in_AMSR2["veg_temp"] + df_S3_pixels_in_AMSR2["veg_std"],
                     color='forestgreen', alpha=0.2)

    # Soil Stats
    ax1.plot(x_idx, df_S3_pixels_in_AMSR2["soil_temp"],
             label=f'Soil T ({soil_s["mean"]:.1f}±{soil_s["std"]:.1f}K)',
             color='saddlebrown', linewidth=2)
    ax1.fill_between(x_idx,
                     df_S3_pixels_in_AMSR2["soil_temp"] - df_S3_pixels_in_AMSR2["soil_std"],
                     df_S3_pixels_in_AMSR2["soil_temp"] + df_S3_pixels_in_AMSR2["soil_std"],
                     color='saddlebrown', alpha=0.2)

    ax1.plot(x_idx, df_S3_pixels_in_AMSR2["tsurf_ka"],
             label=f'Ka TSURF ({tka_s["mean"]:.1f}±{tka_s["std"]:.1f}K)',
             color='red', lw=2)


    if plot_tsurf_adjust:
        ax1.plot(x_idx, df_S3_pixels_in_AMSR2["tsurf_adj"],
                 label=f'Adj TSURF ({tadj_s["mean"]:.1f}±{tadj_s["std"]:.1f}K)',
                 color='magenta', lw=1.5)
    if plot_mpdi:
        ax_mpdi = ax1.twinx()
        ax_mpdi.plot(x_idx, df_S3_pixels_in_AMSR2["mpdi"], color='blue', alpha=0.6)
        ax_mpdi.tick_params(axis='y', labelcolor='blue')
        ax_mpdi.set_ylabel(f'MPDI {mpdi_band}', color='blue')
    if plot_kuka:
        ax_mpdi = ax1.twinx()
        ax_mpdi.plot(x_idx, df_S3_pixels_in_AMSR2["kuka"], color='royalblue', alpha=0.6)
        ax_mpdi.tick_params(axis='y', labelcolor='royalblue')
        ax_mpdi.set_ylabel(f'KuH / KaH', color='royalblue')

    ax1.set_ylabel('Temperature [K]')
    ax1.set_title('Sub-pixel LST per AMSR2 pixel')
    ax1.legend(loc='upper left')

    if not scatter_x: raise ValueError(f"Define x data for scatterplot --> scatter_x = 'veg_temp' or 'soil_temp'" )
    # Left scatter
    x1, y1 = df_S3_pixels_in_AMSR2[scatter_x], df_S3_pixels_in_AMSR2["kuka"]
    s1_stat = usual_stats(x1, y1)

    ax_scatter1 = fig.add_subplot(gs[2, 0])
    ax_scatter1.scatter(x1, y1, alpha=0.6, s=20, c='indigo')
    # lims = [min(x1.min(), y1.min()), max(x1.max(), y1.max())]
    # ax_scatter1.plot(lims, lims, 'r--', label='1:1')
    stats_str = f"r: {s1_stat["r"]:.3f}\nRMSE: {s1_stat["rmse"]:.2f}K\nBias: {s1_stat["bias"]:.2f}K"
    ax_scatter1.text(0.05, 0.95, stats_str, transform=ax_scatter1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_scatter1.set_title(f"tsurf_ka vs {scatter_x}")
    ax_scatter1.set_xlabel(f"{scatter_x}")
    ax_scatter1.set_ylabel("tsurf_ka")

    # Right scatter
    x2, y2 = x1, df_S3_pixels_in_AMSR2["tsurf_adj"]
    s2_stat = usual_stats(x2, y2)

    ax_scatter2 = fig.add_subplot(gs[2, 1])
    ax_scatter2.scatter(x2, y2, alpha=0.6, s=20, c='indigo')
    # ax_scatter2.plot(lims, lims, 'r--', label='1:1')
    stats_str = f"r: {s2_stat["r"]:.3f}\nRMSE: {s2_stat["rmse"]:.2f}K\nBias: {s2_stat["bias"]:.2f}K"
    ax_scatter2.text(0.05, 0.95, stats_str, transform=ax_scatter2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_scatter2.set_title(f"tsurf_adj vs {scatter_x}")
    ax_scatter2.set_xlabel(f"{scatter_x}")
    ax_scatter2.set_ylabel("tsurf_adj")


    plt.suptitle(f"Sentinel-3 & AMSR2 Analysis | {obs_date}", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
