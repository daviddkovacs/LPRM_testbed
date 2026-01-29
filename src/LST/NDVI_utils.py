import glob
import re
from functools import partial
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
from xarray import apply_ufunc
from config.paths import NDVI_path, SLSTR_path
import pandas as pd
from datetime import datetime


def crop2roi(ds,bbox):
    mask = (
            (ds.lon >= bbox[0]) & (ds.lon <= bbox[2]) &
            (ds.lat >= bbox[1]) & (ds.lat <= bbox[3])
    )
    return ds.where(mask, drop=True)


def filternan(array):
    return  array.values.flatten()[~np.isnan(array.values.flatten())]


def clip_swath(ds):
    """
    Some S3 Tiles have a larger (1-2pix) across-scan dim resulting in errors. Thus we crop it.
    """
    return ds.isel(rows=slice(0, 1200))


def filter_empty_var(ds, var = "NDVI"):
    """
    Sometimes NDVI is empty.. then we filter the whole dataset
    """
    valid = ds[var].notnull().any(dim = ["rows","columns"])
    return ds.sel(time=valid)


def open_amsr2(path,
               sensor,
               date_pattern,
               overpass,
               subdir_pattern,
               file_pattern,
               time_start = "2024-01-01",
               time_stop = "2025-01-01",
               ):

    folder = os.path.join(path,sensor,overpass,subdir_pattern,file_pattern)

    files = glob.glob(folder)

    dates_string =  [re.search(date_pattern, p).group(1) for p in files]

    _dates = pd.to_datetime(dates_string)

    date_mask  = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))
    files_valid = np.array(files)[date_mask]

    dataset = xr.open_mfdataset(files_valid,
                                combine ="nested",
                                join = "outer",
                                concat_dim = "time",
                                chunks = "auto").assign_coords(time = _dates[date_mask])

    print(f"Loading dataset finished (AMSR2)")

    return dataset


def open_sltsr(path,
               subdir_pattern,
               date_pattern,
               variable_file,
               # bbox=None,
               georeference_file = "geodetic_in.nc"
               ):

    folder = os.path.join(path,subdir_pattern,variable_file)

    files = glob.glob(folder)

    dates_string =  [(re.search(date_pattern, p).group(1),
                      re.search(date_pattern, p).group(2))for p in files]

    dates_dt = [pd.to_datetime(f"{dt[0]} {dt[1]}") for dt in dates_string]

    dataset = xr.open_mfdataset(files,
                                preprocess =clip_swath,
                                combine ="nested",
                                join = "outer",
                                concat_dim = "time",
                                chunks = "auto").assign_coords(time = dates_dt)

    if georeference_file: # L1 and L2 SLSTR data isnt gridded. lat, lon from external file!

        coord_path = os.path.join(path,subdir_pattern,georeference_file)
        geo_files = glob.glob(coord_path)

        geo = xr.open_mfdataset(geo_files,
                                preprocess =clip_swath,
                                combine="nested",
                                join="outer",
                                concat_dim="time",
                                chunks="auto").assign_coords(time = dates_dt)

        dataset = dataset.assign_coords(
            lat=(("time", "rows", "columns"), geo.latitude_in.data),
            lon=(("time", "rows", "columns"), geo.longitude_in.data)
        )

        print(f"Loading dataset finished ({variable_file})")
    return dataset


def plot_lst(left_da,
             right_da,
             left_params,
             right_params,
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
    plt.show()


def cloud_filtering(dataset,
                    cloud_path=SLSTR_path,
                    cloud_subdir_pattern=f"S3A_SL_2_LST____*",
                    cloud_date_pattern=r'___(\d{8})T(\d{4})',
                    cloud_variable_file="flags_in.nc",
                    threshold = 1):
    """
    Optional cloud masking, with default path and variable parameters to SLSTR cloud flags.
    Strict threshold of 1, filters ALL clouds.
    """

    CLOUD= open_sltsr(path=cloud_path,
                   subdir_pattern=cloud_subdir_pattern,
                   date_pattern=cloud_date_pattern,
                   variable_file=cloud_variable_file,
                        )

    cloudy = xr.where(CLOUD["cloud_in"]>threshold,True,False )

    return xr.where(cloudy, np.nan, dataset)


def snow_filtering(dataset,
                    cloud_path=SLSTR_path,
                    cloud_subdir_pattern=f"S3A_SL_2_LST____*",
                    cloud_date_pattern=r'___(\d{8})T(\d{4})',
                    cloud_variable_file="LST_ancillary_ds.nc",
                    snow_and_ice_flag = 27):
    """
    Optional snow and ice masking, with default path and variable parameters to SLSTR cloud flags.
    """

    SNOWICE= open_sltsr(path=cloud_path,
                   subdir_pattern=cloud_subdir_pattern,
                   date_pattern=cloud_date_pattern,
                   variable_file=cloud_variable_file,
                        )
    snowy = xr.where(SNOWICE["biome"]==27, True, False)

    return xr.where(snowy, np.nan, dataset)
