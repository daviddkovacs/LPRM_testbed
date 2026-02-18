import glob
import os
import re
from typing import Literal, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from LST.datacube_utilities import crop2roi


def assign_time_of_day(dataset, dates):
    """
    Since AMSR2 data has the same date as time dimension for day and night, we need to assign time of day timestamps.
    This works by calculating the mean scantime pixel values over a region of interest.
    This is not exact, but allows for the separation of day and night, and thus merging the two into a single ds.
    :param dataset: cropped dataset to bbox
    :return: dataset with day or night assigned time dims
    """

    mean_scantime_over_bbox = dataset["scantime"].mean(dim =["lat","lon"]).compute()

    base_dates = pd.to_datetime(dates)
    scan_time_deltas = pd.to_timedelta(mean_scantime_over_bbox, unit='s')
    dates_with_scantime = base_dates + scan_time_deltas

    return dataset.assign_coords(time =dates_with_scantime)


def open_amsr2(path,
               sensor,
               overpass,
               date_pattern,
               subdir_pattern,
               file_pattern,
               resolution: Literal["coarse_resolution","medium_resolution"],
               time_start: str = "2024-01-01",
               time_stop: str = "2025-01-01",
               bbox : List[float] = None
               ):

    folder = os.path.join(path,resolution,sensor,overpass,subdir_pattern,file_pattern)

    files = glob.glob(folder)

    dates_string =  [re.search(date_pattern, p).group(1) for p in files]

    _dates = pd.to_datetime(dates_string)

    date_mask  = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))
    files_valid = np.array(files)[date_mask]

    dataset = xr.open_mfdataset(files_valid,
                                combine ="nested",
                                join = "outer",
                                concat_dim = "time",
                                chunks = "auto",
                                decode_timedelta = False)

    dataset_cropped = crop2roi(dataset, bbox)
    dataset_time_of_day  = assign_time_of_day(dataset = dataset_cropped,
                                              dates = _dates[date_mask])

    res_dict = {"coarse_resolution" : 0.25,
                "medium_resolution":  0.1}
    dataset_complete = dataset_time_of_day.assign_attrs(resolution = res_dict[resolution])
    print(f"Loading dataset finished (AMSR2 {overpass})")

    return dataset_complete
