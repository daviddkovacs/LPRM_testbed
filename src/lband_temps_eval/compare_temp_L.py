import glob
import os.path
import numpy as np
import pandas as pd
import re
import xarray as xr
import matplotlib.pyplot as plt
import datetime
from itertools import pairwise
import matplotlib.dates as mdates
import traceback
investigation_period_start = pd.to_datetime("2018-01-01")
investigation_period_end = pd.to_datetime("2019-01-01")

# NIGHT is D--> 0600 at equator
# DAY is A--> 1800 at equator
overpass = "day"
print(overpass)
temporal_match_seconds = "10k"
orbit_direction = {
    "night": "Desc. orbit",
    "day":"Asc. orbit"
}

climers_path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/"
old_temp_path= os.path.join(climers_path, "03_lband_temperatures", "coarse_resolution", "SMAP", f"{overpass}", "20*", f"smap_combined_bt_{overpass}*.nc")

new_temp_path = os.path.join(climers_path, f"07_debug/daytime_retrieval/L_Band_Temps/temps/{temporal_match_seconds}", "coarse_resolution", "SMAP",
                        f"{overpass}", "20*",f"smap_combined_bt_{overpass}*.nc")


def get_multiple_files(path, time_start,time_stop, datepattern = "_(\\d{8})_"):

    files = sorted(glob.glob(path))
    dates_string = [re.search(datepattern, p).group(1) for p in files]
    _dates = pd.to_datetime(dates_string)
    date_mask = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))
    files_valid = np.array(files)[date_mask]

    return files_valid


def get_regression_maps(band):

    path_aux_t = (
        f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/lprm_testing"
        f"/T_aux/{band.upper()}_daytime_LST_regression.nc")

    daytime_stats = xr.open_dataset(path_aux_t)

    return daytime_stats


def timeseries(lat, lon,data):
    _line=  data.sel(lat=lat, lon = lon, method="nearest")
    line= _line.where((150<_line) & (_line<450))
    return line

def _preprocess(ds):
    name = ds.encoding["source"]
    date = re.search(r"(\d{8})", name).group(1)
    date_obj = datetime.datetime.strptime(date, "%Y%m%d")
    ds = ds.assign_coords({"time": [date_obj]})
    return ds

def calc_holmes(data):
    return (data * 0.893 +44.8).compute()

old_files = get_multiple_files(old_temp_path, investigation_period_start, investigation_period_end, datepattern=r"(\d{8})")
new_files = get_multiple_files(new_temp_path, investigation_period_start, investigation_period_end, datepattern=r"(\d{8})")

old_dataset = xr.open_mfdataset(
    old_files,
    combine="nested",
    join="outer",
    concat_dim="time",
    # chunks="auto",
    decode_timedelta=False,
)

new_dataset = xr.open_mfdataset(
    new_files,
    combine="nested",
    join="outer",
    concat_dim="time",
    group = "Ka",
    preprocess=_preprocess,
    decode_timedelta=False,
)


##

old_temp = (old_dataset["bt_36.5V"] * 0.893 + 44.8).compute()
new_temp = (new_dataset["bt_vertical"] * 0.893 + 44.8).compute()

##

daterange = pd.date_range(investigation_period_start,investigation_period_end,freq="10D")

for time_start, time_stop in pairwise(daterange):

    old_mean = old_temp.sel(time=slice(time_start,time_stop)).mean(dim="time")
    new_mean = new_temp.sel(time=slice(time_start,time_stop)).mean(dim="time")

    difference_mean = new_mean- old_mean

    # img1 = old_mean.plot.imshow(figsize=(20,10), vmin=250, vmax=320)
    # plt.title(f"Old T (AMSR2, FY3D, GMI) \nSMAP {orbit_direction[overpass]} tolerance (s): {temporal_match_seconds} \n{time_start} -- {time_stop}")
    # plt.show()
    #
    # img2 = new_mean.plot.imshow(figsize=(20,10), vmin=250, vmax=320)
    # plt.title(f"NEW T (WS, SSMIS F16,17,18) \nSMAP {orbit_direction[overpass]} tolerance (s): {temporal_match_seconds} \n{time_start} -- {time_stop}")
    # plt.show()

    img3 = difference_mean.plot.imshow(figsize=(20,10), vmin=-10, vmax=10, cmap="coolwarm")
    plt.title(f"new_mean  - old_mean  \nSMAP {orbit_direction[overpass]} tolerance (s): {temporal_match_seconds} \n{time_start} -- {time_stop}")
    plt.show()

##
lc_lut = {
"sahara_desert" :(20.30, 8.76),
"vienna_mixed" : (47.96, 15.343),
"india_agri" : (22.44, 76.33),
"cornbelt_agri" : (45.98, -93.33),
"midwest_arid" : (32.264, -108.0),
"siberia_decid_broadleaf" : (65, 91),
"yakutsk_broadleaf" : (62,133),
"w_greenland" : (67., -51.14),
"e_australia_desert" : (-19.74, 124.11),
"n_alaska_tundra" : (68.91, -156.60),
"eswatini_savanna" :( -24.990,30.93),
"s_patagonia" : ( -52.051187, -73.276493),
}

def bias(x,y):
    x_mean = x.mean()
    y_mean = y.mean()
    return (x_mean - y_mean).item()

def rmse(x, y):
    return np.sqrt(((x - y)**2).mean()).item()

def r2_score(x, y):
    df = pd.DataFrame({'x': x, 'y': y})
    return df['x'].corr(df['y'])


swath_files = "/home/ddkovacs/Desktop/L_Band_Temps/debug/planet/*.nc"

files = glob.glob(swath_files)

swaths = xr.open_mfdataset(files,
                           combine="nested",
                           join="outer",
                           concat_dim="time",
                           decode_timedelta=False,
                           engine = "netcdf4",
                           preprocess=_preprocess).sortby('time')

afternoon_bt = swaths.sel(swath="130PM").sel(time=  slice(investigation_period_start,investigation_period_end))
next_night_bt = swaths.sel(swath="130AM_nextday").sel(time= slice(investigation_period_start,investigation_period_end))

afternoon = calc_holmes(afternoon_bt)
next_night = calc_holmes(next_night_bt)

for _region,latlon_tuple in lc_lut.items():

    region = _region
    lat = latlon_tuple[0]
    lon = latlon_tuple[1]

    old_temp_line = timeseries(lat, lon, old_temp)
    new_temp_line = timeseries(lat, lon, new_temp)

    fig, ax1 = plt.subplots(figsize=(16, 7))

    # Calculate the difference
    dif_temp  = new_temp_line - old_temp_line

    # --- Primary Axis (ax1) for Absolute Temperatures ---
    color1 = 'black'
    old_temp_line.dropna(dim='time').sortby('time').plot(ax=ax1, label="Old Planet T", color="tab:blue")
    new_temp_line.dropna(dim='time').sortby('time').plot(ax=ax1, label="New T", color="tab:orange")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.set_ylim([230, 330])
    ax1.set_ylabel("T [K]", color=color1)
    ax1.set_xlabel(f"Date ({overpass} overpass AMSR2)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', labelcolor=color1)

    color2 = 'tab:green'
    ax2 = ax1.twinx()

    dif_temp.dropna(dim='time').sortby('time').plot(ax=ax2, label="New - Old (planet) dif", color=color2, linestyle="--")

    ax2.set_ylim([-40, 40])  # Adjust limits as necessary for your difference data
    ax2.set_ylabel("Δ T [K]", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2, color=color2)
    ax2.spines['right'].set_color(color2)

    # --- Statistics Text Box ---
    val_bias = bias(old_temp_line, new_temp_line)
    val_rmse = rmse(old_temp_line, new_temp_line)
    val_r2 = r2_score(old_temp_line, new_temp_line)

    stats_text = (
        f"Bias (New-old(planet)): {val_bias:.2f} K\n"
        f"RMSE: {val_rmse:.2f} K\n"
        f"R: {val_r2:.2f}"
    )

    ax1.text(
        x=0.05,
        y=0.95,
        s=stats_text,
        transform=ax1.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # --- Combine Legends and Finalize Layout ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # Place legend outside the text box to avoid overlap (e.g., lower right or upper right)
    ax1.set_title("")
    ax2.set_title("")
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

    fig.suptitle(f"New-old(planet) Temps \n{region} lat: {lat}, lon: {lon}", fontsize=14)
    plt.tight_layout()
    plt.show()


    #planet
    try:
        fig, ax1 = plt.subplots(figsize=(16, 7))

        afternoon_ts = timeseries(lat, lon, afternoon)["combined_temperatures"]
        next_night_ts = timeseries(lat, lon, next_night)["combined_temperatures"]
        # mean = (afternoon_ts + next_night_ts) / 2
        dif_planet = afternoon_ts - next_night_ts

        color1 = 'black'

        afternoon_ts.dropna(dim='time').plot(ax=ax1, label="130PM", color="tab:blue")
        next_night_ts.dropna(dim='time').plot(ax=ax1, label="130AM_nextday", color="tab:orange")

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.set_ylim([230, 330])
        ax1.set_xlabel(f"Date ({overpass} overpass AMSR2)")
        ax1.tick_params(axis='x', rotation=45)

        # Color the primary y-axis label and ticks
        ax1.set_ylabel("T [K]", color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # --- Secondary Axis (ax2) for the Difference ---
        color2 = 'tab:red'
        ax2 = ax1.twinx()

        dif_planet.dropna(dim='time').plot(ax=ax2, label="aft-nextday dif", color=color2, linestyle="--")

        ax2.set_ylim([-40, 40])

        # Color the secondary y-axis label, ticks, and the spine (the right border line)
        ax2.set_ylabel("Δ T [K]", color=color2)
        ax2.tick_params(axis='y', labelcolor=color2, color=color2)
        ax2.spines['right'].set_color(color2)

        # --- Combine Legends ---
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        ax1.set_title("")
        ax2.set_title("")
        fig.suptitle(f"afternoon-nextday dif.\n{region} lat: {lat}, lon: {lon}", fontsize=14)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"{_region} not valid")
        print(traceback.format_exc())


##


for d in daterange:
    _aft_obs = afternoon.sel(time=d, method="nearest")["combined_temperatures"]
    _next_obs = next_night.sel(time=d, method="nearest")["combined_temperatures"]
    _diff = _aft_obs - _next_obs

    vmin = 250
    vmax = 300

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(nrows=2, ncols=2)

    ax_aft = fig.add_subplot(gs[0, 0])  # Top-Left
    ax_next = fig.add_subplot(gs[0, 1], sharey=ax_aft)  # Top-Right (shares Y with top-left)
    ax_diff = fig.add_subplot(gs[1, :])  # Bottom (spans all columns in row 1)

    # 3. Plot Afternoon
    _aft_obs.plot(
        ax=ax_aft,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )
    ax_aft.set_title("13:30 (afternoon) overpass")

    # 4. Plot Next Night (with colorbar for the top row)
    _next_obs.plot(
        ax=ax_next,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={"label": "LST"}
    )
    ax_next.set_title("01:30 (next night) overpass")


    max_abs_diff = np.abs(_diff).max().item()

    _diff.plot(
        ax=ax_diff,
        cmap="RdBu_r",
        vmin=40,
        vmax=-40,
        add_colorbar=True,
        cbar_kwargs={"label": "LST"}
    )
    ax_diff.set_title("Difference (Afternoon - Next Night)")

    plt.suptitle(f"Planet's LST composite: GMI, AMSR2, FY3D\nDate: {d}", fontsize=16)
    plt.tight_layout()
    plt.show()