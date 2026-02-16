import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

from LST.datacube_utilities import clip_swath, crop2roi, clean_pad_data, filter_empty_var


def open_slstr_date(lst,
              anc,
              cloud,
              coord,
              day):

    LST = xr.open_mfdataset(lst,
                            chunks="auto",
                            decode_timedelta = False,
                            join= "outer",
                            combine="nested",
                            preprocess=clip_swath,
                            ).assign_coords(time = day)[["LST"]]

    ANC = xr.open_mfdataset(anc,
                            chunks="auto",
                            decode_timedelta = False,
                            join= "outer",
                            combine="nested",
                            preprocess=clip_swath,
                            ).assign_coords(time = day)[["NDVI","biome"]]
    NDVI = ANC["NDVI"]

    SNOWICE = ANC["biome"]
    CLOUD = xr.open_mfdataset(cloud,
                              chunks="auto",
                              decode_timedelta = False,
                              join= "outer",
                              combine="nested",
                              preprocess=clip_swath,
                              ).assign_coords(time = day)["bayes_in"]

    COORDS = xr.open_mfdataset(coord,
                               chunks="auto",
                               decode_timedelta = False,
                               join= "outer",
                               combine="nested",
                               preprocess=clip_swath,
                               ).assign_coords(time = day)[["latitude_in","longitude_in"]]

    DATA = xr.merge([NDVI,LST])[["LST","NDVI"]]

    DATA = DATA.assign_coords(
        lat=(( "rows", "columns"), COORDS.latitude_in.values),
        lon=(( "rows", "columns"), COORDS.longitude_in.values)
    )


    cloudy = xr.where(CLOUD == 2, True, False) # TODO
    CLOUD_FILTERED = xr.where(cloudy, np.nan, DATA)

    snowy = xr.where(SNOWICE==27, True, False)
    CLOUD_SNOW_FILTERED = xr.where(snowy, np.nan, CLOUD_FILTERED)

    print(f"Succesfully read SLSTR: {day}")

    return CLOUD_SNOW_FILTERED


def open_sltsr(path,
               bbox,
               subdir_pattern=f"S3?_SL_2_LST____*",
               date_pattern=r'___(\d{8})T(\d{4})',
               time_start="2024-01-01",
               time_stop="2025-01-01",
               ):
    """
    Sorry to all my colleauges and to people who like elegant code, I also do.
    However, reading SLSTR files, with multi-year timespan in an elegant way, did not work.
    """
    folder_lst = os.path.join(path,subdir_pattern,"LST_in.nc")
    folder_anc = os.path.join(path,subdir_pattern,"LST_ancillary_ds.nc")
    folder_cloud = os.path.join(path,subdir_pattern,"flags_in.nc")
    coord_path = os.path.join(path,subdir_pattern,"geodetic_in.nc")

    files_lst = glob.glob(folder_lst)
    files_anc = glob.glob(folder_anc)
    files_cloud = glob.glob(folder_cloud)
    geo_files = glob.glob(coord_path)

    dates_string =  [(re.search(date_pattern, p).group(1),
                      re.search(date_pattern, p).group(2))for p in files_lst] # could be also files_ndvi (date comes from fname)

    _dates = pd.to_datetime([f"{dt[0]} {dt[1]}" for dt in dates_string])

    date_mask  = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))

    files_valid_lst = np.array(files_lst)[date_mask]  # LST
    files_valid_anc = np.array(files_anc)[date_mask]  # Ancillary (NDVI, Snow and Ice flags)
    files_valid_cloud = np.array(files_cloud)[date_mask]  # Cloud classification
    geo_files_valid = np.array(geo_files)[date_mask] # COORDS
    dates_valid = np.array(_dates)[date_mask] # days

    big_data = []

    for lst, anc, cloud, coord, day in zip(files_valid_lst,
                                           files_valid_anc,
                                           files_valid_cloud,
                                           geo_files_valid,
                                           dates_valid):

        daily_da = open_slstr_date(lst, anc, cloud, coord, day)
        cropped_daily_da=  crop2roi(daily_da, bbox=bbox)
        big_data.append(cropped_daily_da)

    padded_data = clean_pad_data(big_data, x = "rows", y = "columns")

    _dataset = xr.concat(padded_data, dim = "time")
    _dataset = filter_empty_var(_dataset)

    return _dataset.sortby("time")
