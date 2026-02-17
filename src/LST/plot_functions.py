import numpy as np
import pandas as pd
import matplotlib.patches as patches
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from LST.datacube_utilities import subset_statistics

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


def amsr2_lst_figure(ds,
               plot_params):

    lon_min = np.min(ds.lon.values)
    lon_max = np.max(ds.lon.values)
    lat_min = np.min(ds.lat.values)
    lat_max = np.max(ds.lat.values)

    res = ds.attrs.get("resolution", 0.25)
    extent = [lon_min - res / 2, lon_max + res / 2, lat_min - res / 2, lat_max + res / 2]

    plt.figure()

    ds.plot.imshow(
        cmap=plot_params["cmap"],
        vmin=plot_params["vmin"],
        vmax=plot_params["vmax"],
        extent=extent
    )

    date_str = ds.time.dt.strftime('%Y-%m-%d').item()
    plt.title(f"AMSR2 LST in bounding box\n{date_str}")
    plt.show()


def usual_stats(x,y):
    r =x.corr(y)
    bias = (y - x).mean()
    rmse = np.sqrt(((y - x) ** 2).mean())
    return {"r" : r , "bias" : bias , "rmse" : rmse}


def boxplot_timeseries(df,mpdi_band=""):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    unique_dates = df['time'].unique()
    date_nums = mdates.date2num(unique_dates)

    def get_grouped_data(col_name):
        grouped_data = []
        for d in unique_dates:
            day_data = df.loc[df['time'] == d, col_name].dropna().values

            if len(day_data) > 0 and isinstance(day_data[0], (list, np.ndarray)):
                merged_pixels = np.hstack(day_data)
                grouped_data.append(merged_pixels)
            else:
                grouped_data.append(day_data)
        return grouped_data

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 12), sharex=True)

    # offset of bars
    width = 1.0

    # Soil (Left)
    soil_data = get_grouped_data('soil_array')
    bp_soil = ax1.boxplot(soil_data,
                          positions=date_nums ,
                          widths=width,
                          patch_artist=True, boxprops=dict(facecolor='#8c564b', alpha=0.8),
                          medianprops=dict(color='black'), showfliers=False)

    # Veg (Center)
    veg_data = get_grouped_data('veg_array')
    bp_veg = ax1.boxplot(veg_data,
                         positions=date_nums,
                         widths=width,
                         patch_artist=True, boxprops=dict(facecolor='#2ca02c', alpha=0.8),
                         medianprops=dict(color='black'), showfliers=False)

    # Ka Temp (Right)
    ka_data = get_grouped_data('tsurf_ka')
    bp_ka = ax1.boxplot(ka_data,
                        positions=date_nums,
                        widths=3,
                        patch_artist=True, boxprops=dict(facecolor='red', alpha=0.6),
                        medianprops=dict(color='darkred',linewidth=3), showfliers=False)

    try:
        ax1.legend([bp_soil["boxes"][0], bp_veg["boxes"][0], bp_ka["boxes"][0]],
                   ['Soil Temp', 'Veg Temp', 'Ka Temp'], loc='upper right')
    except IndexError:
        pass

    ax1.set_ylabel("T [K]", fontweight='bold')
    ax1.set_title("Temperatures", loc='left', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    kuka_data = get_grouped_data('kuka')
    ax2.boxplot(kuka_data, positions=date_nums, widths=0.6,  # Wider since it's alone
                patch_artist=True, boxprops=dict(facecolor='#9467bd', alpha=0.7),
                medianprops=dict(color='black'), showfliers=False)

    ax2.set_ylabel("Index", color='#9467bd', fontweight='bold')
    ax2.set_title("KuKa", loc='left', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    mpdi_data = get_grouped_data('mpdi')
    ax3.boxplot(mpdi_data, positions=date_nums, widths=0.6,
                patch_artist=True, boxprops=dict(facecolor='#1f77b4', alpha=0.7),
                medianprops=dict(color='black'), showfliers=False)

    ax3.set_ylabel("MPDI", color='#1f77b4', fontweight='bold')
    ax3.set_title(f"{mpdi_band.upper()} MPDI", loc='left', fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)

    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    pad = pd.Timedelta(days=1)
    ax3.set_xlim(df['time'].min() - pad, df['time'].max() + pad)

    plt.tight_layout()
    return fig




def plot_hexbin(df, x_col, y_col, xlim = [273, 325], ylim=[273, 325]):

    x = df[x_col]
    y = df[y_col]
    stats = usual_stats(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))

    hb = ax.hexbin(x, y,
                   gridsize=100, cmap='inferno', mincnt=1)

    ax.plot(xlim, ylim, 'k--', alpha=0.8, linewidth=1, zorder=10)

    textstr = '\n'.join((
        f'$R = {stats["r"]:.2f}$',
        f'$RMSE = {stats["rmse"]:.2f}$ K',
        f'$Bias = {stats["bias"]:.2f}$ K',
        f'$N = {len(x)}$'
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}')


    plt.show()


def combined_validation_dashboard(LST_L1B,
                                  NDVI_L1B,
                                  df_S3_pixels_in_AMSR2,
                                  bbox=None,
                                  plot_mpdi=False,
                                  plot_tsurf_adjust = False,
                                  plot_kuka=False,
                                  mpdi_band=None,
                                  scatter_x = None,
                                  LST_params = LST_plot_params,
                                  NDVI_params = NDVI_plot_params,
                                  ):
    """
    Combines spatial plots, pixel-wise time series, and 1:1 scatter validation.
    """
    LST_L1B = LST_L1B.isel(time=0) if 'time' in LST_L1B.dims and LST_L1B.sizes['time'] > 1 else LST_L1B.squeeze()
    NDVI_L1B = NDVI_L1B.isel(time=0) if 'time' in NDVI_L1B.dims and NDVI_L1B.sizes['time'] > 1 else NDVI_L1B.squeeze()
    LST_L1B = LST_L1B.dropna(dim='rows', how='all').dropna(dim='columns', how='all')
    NDVI_L1B = NDVI_L1B.dropna(dim='rows', how='all').dropna(dim='columns', how='all')
    # --- NEW FIX ENDS HERE ---
    obs_date = pd.to_datetime(LST_L1B.time.values).strftime('%Y-%m-%d')

    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(3, 2)

    ax_lst = fig.add_subplot(gs[0, 0])
    LST_L1B.plot.pcolormesh(x=LST_params["x"], y=LST_params["y"], ax=ax_lst, cmap=LST_params["cmap"],
                            vmin=LST_params["vmin"], add_colorbar=True, cbar_kwargs=LST_params["cbar_kwargs"])
    ax_lst.set_title(LST_params["title"])

    ax_ndvi = fig.add_subplot(gs[0, 1])
    NDVI_L1B.plot.pcolormesh(x=NDVI_params["x"], y=NDVI_params["y"], ax=ax_ndvi, cmap=NDVI_params["cmap"],
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



def fill_plot_coords(ds_slice):
    """Forward and backward fills lat/lon NaNs to prevent pcolormesh errors."""
    for coord in ["lat", "lon"]:
        ds_slice[coord] = (ds_slice[coord]
                           .ffill(dim="column").bfill(dim="column")
                           .ffill(dim="row").bfill(dim="row"))
    return ds_slice

def plot_modis_comparison(ndvi_da, lst_da, ndvi_time=4, lst_time=8):
    """Slices, fixes coordinates, and plots NDVI and LST side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ndvi_slice = fill_plot_coords(ndvi_da.sel(time=ndvi_time, method="nearest").copy())
    ndvi_slice.plot(ax=ax1, cmap="RdYlGn", x="lon", y="lat")
    ax1.set_title(f"MODIS NDVI (Time Index: {ndvi_slice.time.values})")

    lst_slice = fill_plot_coords(lst_da.sel(time=lst_time, method="nearest").copy())
    lst_slice.plot(ax=ax2, cmap="inferno", x="lon", y="lat")
    ax2.set_title(f"MODIS LST (Time Index: {lst_slice.time.values})")

    plt.tight_layout()
    plt.show(block=True)