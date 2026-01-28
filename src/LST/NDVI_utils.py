import glob
import re
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
from xarray import apply_ufunc
from config.paths import NDVI_path, SLSTR_path
import pandas as pd
from datetime import datetime


def crop2roi(ds,bbox):
    if bbox:
        if "latitude" in ds.coords:
            return ds[""]
        return ds.sel(latitude=slice(bbox[3], bbox[1]),longitude=slice(bbox[0], bbox[2]))
    else:
        return ds


def filter_empty(ds, var = "NDVI"):

    valid = ds[var].notnull().any(dim = ["rows","columns"])
    return ds.sel(time=valid)


def open_sltsr(path,
               subdir_pattern,
               date_pattern,
               variable_file,
               bbox=None,
               georeference_file = "geodetic_in.nc"
               ):

    folder = os.path.join(path,subdir_pattern,variable_file)

    files = glob.glob(folder)

    dates_string =  [(re.search(date_pattern, p).group(1),
                      re.search(date_pattern, p).group(2))for p in files]

    dates_dt = [pd.to_datetime(f"{dt[0]} {dt[1]}") for dt in dates_string]

    dataset = xr.open_mfdataset(files,
                                # preprocess =partial(crop2roi, bbox =bbox),
                                combine ="nested",
                                join = "outer",
                                concat_dim = "time",
                                chunks = "auto").assign_coords(time = dates_dt)

    if georeference_file: # L1 and L2 SLSTR data isnt gridded. lat, lon from external file!

        coord_path = os.path.join(path,subdir_pattern,georeference_file)
        geo_files = glob.glob(coord_path)

        geo = xr.open_mfdataset(geo_files,
                                combine="nested",
                                join="outer",
                                concat_dim="time",
                                chunks="auto").assign_coords(time = dates_dt)

        dataset = dataset.assign_coords(
            lat=(("time", "rows", "columns"), geo.latitude_in.data),
            lon=(("time", "rows", "columns"), geo.longitude_in.data)
        )

    return dataset