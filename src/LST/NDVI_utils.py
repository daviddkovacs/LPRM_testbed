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


def clip_before(ds):
    return ds.isel(rows=slice(0, 1200))

def filter_empty(ds, var = "NDVI"):
    """
    Sometimes NDVI is empty.. then we filter the whole dataset
    """
    valid = ds[var].notnull().any(dim = ["rows","columns"])
    return ds.sel(time=valid)


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
                                preprocess =clip_before,
                                combine ="nested",
                                join = "outer",
                                concat_dim = "time",
                                chunks = "auto").assign_coords(time = dates_dt)

    if georeference_file: # L1 and L2 SLSTR data isnt gridded. lat, lon from external file!

        coord_path = os.path.join(path,subdir_pattern,georeference_file)
        geo_files = glob.glob(coord_path)

        geo = xr.open_mfdataset(geo_files,
                                preprocess =clip_before,
                                combine="nested",
                                join="outer",
                                concat_dim="time",
                                chunks="auto").assign_coords(time = dates_dt)

        dataset = dataset.assign_coords(
            lat=(("time", "rows", "columns"), geo.latitude_in.data),
            lon=(("time", "rows", "columns"), geo.longitude_in.data)
        )

        # if bbox:
        #     dataset = crop2roi(dataset,bbox)

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
