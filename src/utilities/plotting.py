import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import os
import xarray as xr
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
from utilities.utils import pearson_corr

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


def plot_maps_LPRM(ds,
                   cbar_lut,
                   date,
                   ):

    ncols = len(cbar_lut.keys()) if (len(cbar_lut.keys()) <3) else 3
    nrows = 1 if len(cbar_lut.keys()) <= 3 else 2

    fig, axes = plt.subplots(nrows,ncols, figsize=(10, 6))
    fig.suptitle(date)

    axes = axes.flatten()

    for ax, var in zip(axes, cbar_lut.keys()):
        color = "RdYlBu" if "DIF" in var else "viridis"
        da = ds[var]
        da.plot(ax=ax, vmin=cbar_lut[var][0], vmax=cbar_lut[var][1], cmap=color, add_colorbar=True)
        ax.set_title(var)
        ax.axis('off')
        def format_coord(x, y, da=da, var=var):
            sel = ds.sel({"LON": x, "LAT": y}, method="nearest")
            out = []
            for v in list(cbar_lut.keys()):
                out.append(f"{v}={float(sel[v]):.3f}")
            return f"lon={x:.3f}, lat={y:.3f}, " + ", ".join(out)
        ax.format_coord = format_coord
    plt.tight_layout()
    plt.show()


def plot_maps_day_night(
        merged_df,
        night_LPRM,
        sat_band,
):
    """
    Plot maps from Night LPRM from day (adjusted) lprm and their difference
    :param merged_df: Merged df containing adjusted daytime retrievals
    :param night_LPRM: Import night retrievals, untouched.
    :param sat_band: Band
    :return:
    """

    night_data = night_LPRM[f"SM_{sat_band}"]
    original_data = merged_df[f"SM_{sat_band}"]
    adj_data = merged_df[f"SM_ADJ"]

    night_trim = night_data.isel(LON=slice(0, adj_data.sizes['LON']),
                                 LAT=slice(0, adj_data.sizes['LAT']))

    night_trim = night_trim.sortby(["LAT", "LON"])
    adj_data = adj_data.sortby(["LAT", "LON"])

    diff_values = np.where(
        np.isnan(adj_data.data) | np.isnan(night_trim.data),
        np.nan,
        night_trim.data - adj_data.data
    )

    diff =  xr.DataArray(data = diff_values,
                         dims = ["LAT", "LON"],
                         coords =  dict(
                             LAT =adj_data["LAT"],
                             LON =adj_data["LON"],
                         ))

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

    night_trim.plot.pcolormesh(
        x="LON", y="LAT", cmap="viridis", ax=axes[0,0], vmin= 0, vmax= 0.5,
    )
    axes[0,0].set_title(f"Night SM_{sat_band}")

    adj_data.plot.pcolormesh(
        x="LON", y="LAT", cmap="viridis", ax=axes[0,1], vmin= 0, vmax= 0.5,
    )
    axes[0,1].set_title(f"Day Adjusted SM_{sat_band}")

    diff.plot.pcolormesh(
        x="LON", y="LAT", cmap="coolwarm", ax=axes[1,0], vmin= -0.8, vmax= 0.8,
    )
    axes[1,0].set_title("Night − Day")

    axes[1,1].hist(diff.values.flatten(), bins = 50, range= (-0.5, 0.5), histtype = 'bar' , color = "tab:blue")
    axes[1,1].set_title("Night − Day")

    plt.show()


def plot_timeseries(dt_ori_ds, dt_adj_ds, nt_ds,lat,lon,sat_band = None):

    day_orig = dt_ori_ds.sel(LAT=lat, LON=lon, method="nearest")
    day_adjust = dt_adj_ds.sel(LAT=lat, LON=lon, method="nearest")
    night_array = nt_ds.sel(LAT=lat, LON=lon, method="nearest")

    day_adjust.plot(label = "Day adjust")
    day_orig.plot(label = "Day original")
    night_array.plot(label = "night")
    plt.legend()
    plt.plot()
    # bias_adjust = night_array.mean() - day_adjust.mean()
    # bias_orig = night_array.mean() - day_orig.mean()
    #
    # r_night_adj = pearson_corr(night_array, f"SM_{sat_band}",
    #                            day_adjust,"SM_ADJ")
    #
    # r_night_orig = pearson_corr(night_array, f"SM_{sat_band}",
    #                            day_orig,f"SM_{sat_band}")



def temp_sm_plot(
    insitu_t, satellite_t, satellite_t_soil, satellite_t_canopy,
    insitu_sm, satellite_sm, satellite_adj, **kwargs):

    insitu_t["soil_temperature"] = insitu_t["soil_temperature"] +273.15
    insitu_sm_series = insitu_sm.iloc[:, 0]
    satellite_sm_series = satellite_sm.to_series()
    satellite_adj_series = satellite_adj.to_series()

    df = pd.concat(
        [
            insitu_sm_series,
            satellite_sm_series,
            satellite_adj_series,
        ],
        axis=1
    ).dropna()
    df.columns = ["insitu", "lprm", "lprm_adj",]

    r_sat = df["insitu"].corr(df["lprm"])
    r_adj = df["insitu"].corr(df["lprm_adj"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    insitu_t.plot(ax=ax, label="ISMN T_Soil")
    satellite_t.plot(ax=ax, label="TSURF")
    satellite_t_soil.plot(ax=ax, label="T_soil_hull")
    satellite_t_canopy.plot(ax=ax, label="T_canopy_hull")
    ax.set_title("Temperature")
    ax.legend()

    ax = axes[1]
    insitu_sm.plot(ax=ax, label="ISMN SM")
    satellite_sm.plot(ax=ax, label="LPRM SM (normal)")
    satellite_adj.plot(ax=ax, label="LPRM SM (Adjusted!)")
    ax.set_title("Soil Moisture")
    ax.legend()

    ax.text(
        0.01, 0.97,
        f"r(insitu, sat) = {r_sat:.2f}\n"
        f"r(insitu, adj) = {r_adj:.2f}",
        transform=ax.transAxes,
        va="top"
    )

    t = satellite_sm.indexes["time"]
    axes[1].set_xlim(t.min(), t.max())

    if kwargs:
        plt.suptitle(
            f"{kwargs.get('name')}\nlat: {kwargs.get('lat')}, lon: {kwargs.get('lon')}",
            fontsize=16
        )

    plt.tight_layout()
    plt.show()