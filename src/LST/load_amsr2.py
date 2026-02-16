import glob
import os
import re
from typing import Literal, List

import numpy as np
import pandas as pd
import xarray as xr

from LST.datacube_utilities import crop2roi


def open_amsr2(path,
               sensor,
               overpass,
               date_pattern,
               subdir_pattern,
               file_pattern,
               resolution: Literal["coarse_resolution","medium_resolution"],
               time_start = "2024-01-01",
               time_stop = "2025-01-01",
               bbox = List[float]
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
                                decode_timedelta = False).assign_coords(time = _dates[date_mask])

    res_dict = {"coarse_resolution" : 0.25,
                "medium_resolution":  0.1}
    dataset = dataset.assign_attrs(resolution = res_dict[resolution])
    print(f"Loading dataset finished (AMSR2)")

    return crop2roi(dataset, bbox)
