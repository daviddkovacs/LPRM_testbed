import glob
import os.path
import numpy as np
import pandas as pd
import re
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

time_start = "2020-01-01"
time_stop = "2021-01-01"
overpass = "day"
band_current = "ku"
print(band_current)

climers_path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/"
bt_path = os.path.join(climers_path, "01_resampled_bt", "coarse_resolution", "AMSR2", f"{overpass}", "20*","amsr2_l1bt*.nc")


def get_multiple_files(path, time_start,time_stop, pattern = "_(\\d{8})_"):

    files = sorted(glob.glob(path))
    dates_string = [re.search(pattern, p).group(1) for p in files]
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
    line= _line.where((280<_line) & (_line<400))
    return line


files = get_multiple_files(bt_path, time_start, time_stop)
aux_t_maps = get_regression_maps(band_current)


dataset = xr.open_mfdataset(
    files,
    combine="nested",
    join="outer",
    concat_dim="time",
    # chunks="auto",
    decode_timedelta=False,
)

ka_bt = dataset["bt_36.5V"]
t_holmes = (ka_bt * 0.893 + 44.8).compute()
holmes_mean = t_holmes.mean(dim="time")

slope, intercept  = aux_t_maps["slope"], aux_t_maps["intercept"]
t_regression = (ka_bt * slope + intercept).compute()
regression_mean = t_regression.mean(dim="time")

difference =  t_regression - t_holmes
difference_mean = difference.mean(dim="time")


## Temporal mean map
holmes_mean.plot.imshow(figsize=(20,10), vmin=280, vmax=350)
plt.title(f"Holmes T mean {time_start} -- {time_stop}")
plt.show()

regression_mean.plot.imshow(figsize=(20,10), vmin=280, vmax=350)
plt.title(f"Regression {band_current} T mean {time_start} -- {time_stop}")
plt.show()

difference_mean.plot.imshow(figsize=(20,10),  vmin=-10, vmax=10, cmap ="coolwarm")
plt.title(f"Dif. (RegressionT - HolmesT {band_current}) {time_start} -- {time_stop}")
plt.show()

##
lc_lut = {
    "sahara_desert" :(20.30, 8.76),
    "vienna_mixed" : (47.96, 15.343),
    "india_agri" : (22.44, 76.33),
    "cornbelt_agri" : (45.98, -93.33),
    "midwest_arid" : (32.264, -108.0),
    "siberia_decid_broadleaf" : (63.9, 78.714),
    "w_australia_mixed" : (-30.86, 151.0),
    "e_australia_desert" : (-19.74, 124.11),
    "n_alaska_tundra" : (68.91, -156.60),
    "eswatini_savanna" :( -24.990,30.93)
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


for _region,latlon_tuple in lc_lut.items():

    region = _region
    lat = latlon_tuple[0]
    lon = latlon_tuple[1]

    t_holmes_line = timeseries(lat,lon,t_holmes)
    t_regression_line = timeseries(lat,lon,t_regression)

    fig, ax = plt.subplots()

    t_holmes_line.plot(ax=ax,label ="Holmes T")
    t_regression_line.plot(ax=ax,label = f"Regression T ({band_current.upper()})")

    val_bias = bias(t_holmes_line, t_regression_line)
    val_rmse = rmse(t_holmes_line, t_regression_line)
    val_r2 = r2_score(t_holmes_line, t_regression_line)

    # Format the text block with newlines (\n)
    stats_text = (
        f"Bias (Mean holmes-Mean reg.): {val_bias:.2f} K\n"
        f"RMSE: {val_rmse:.2f} K\n"
        f"R: {val_r2:.2f}"
    )

    plt.title(f"{region} lat: {lat}, lon: {lon}")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.text(
        x=0.05,
        y=0.95,
        s=stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylim([280,340])
    plt.ylabel("T [K]")
    plt.xlabel(f"Date ({overpass} overpass AMSR2)")
    plt.tight_layout()
    plt.show()

    x = 1