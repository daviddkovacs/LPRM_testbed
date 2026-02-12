import glob
import re
import matplotlib

from LST.datacube_utilities import crop2roi, clip_swath, filter_empty_var, get_edges, clean_pad_data

matplotlib.use("TkAgg")
import numpy as np
import xarray as xr
import os
from typing import Literal, List
import pandas as pd
from config.paths import S3_SLSTR_path, path_bt, MODIS_path


# ---------------------------------------
# DATACUBE PROCESSORS
def temporal_subset_dc(SLSTR, AMSR2, date):
    """
    Select the closest date to SLSTR, and thus select this date to access AMSR2
    """
    # SLSTR["time"] = SLSTR.time.sortby("time")
    SLSTR = SLSTR.drop_duplicates(dim="time")
    SLSTR_obs = SLSTR.sel(time=date, method="nearest")

    # We select SLSTR's observation to get AMSR2. the frequency of obs for AMSR2 is higher.
    AMSR2_obs = AMSR2.sortby('time').sel(time=SLSTR_obs.time.dt.floor("d"), method="nearest")

    return {"SLSTR": SLSTR_obs, "AMSR2": AMSR2_obs}


def spatial_subset_dc(SLSTR, AMSR2,  bbox):
    """
    SLSTR is cut to the full spatial extent of AMSR2.
    Both AMSR2 and SLSTR cropped to bbox
    """
    AMSR2 = crop2roi(AMSR2, bbox)
    res = AMSR2.attrs["resolution"]

    AMSR2_bbox = [get_edges(AMSR2.lon.values, res).min(),
                  get_edges(AMSR2.lat.values, res).min(),
                  get_edges(AMSR2.lon.values, res).max(),
                  get_edges(AMSR2.lat.values, res).max()]

    SLSTR_roi = crop2roi(SLSTR, AMSR2_bbox)

    return {"SLSTR": SLSTR_roi, "AMSR2": AMSR2}


def OPTI_AMSR2_DATACUBES(region :Literal["sahel", "siberia", "midwest", "ceu"],
                          bbox= List[float],
                          sensor= Literal["MODIS", "SLSTR"],
                           AMSR2_path = path_bt,
                           time_start = "2024-01-01",
                           time_stop = "2025-01-01",
                           ):
    """
    Main function to obtain SLSTR and AMSR2 observations, cut to the ROI.
    :param date: Date
    :param bbox: Bound box (lonmin, latmin, lonmax, latmax)
    :param optical_path: Path where SLSTR data is stored. Accepts "SL_2_LST*.SEN3" unpacked folders.
    :param AMSR2_path: Path where AMSR2 brightness temperatures are stored
    :param region: Region of SLSTR. Currently downloaded: Sahel, Siberia and US Midwest
    :return: dictionary with SLSTR and AMSR2 datacubes.
    """

    SLSTR_path_region = os.path.join(S3_SLSTR_path,region)

    if sensor.upper() == "SLSTR":

        optcial_stack = open_sltsr(SLSTR_path_region,
                                 time_start = time_start,
                                 time_stop = time_stop,
                                 bbox=bbox
                                 )

    elif sensor.upper() == "MODIS":

        optical_stack = open_modis(MODIS_path, bbox=bbox, subdir_pattern="reflectance", time_start="2019-01-01",
                                   time_stop="2020-01-01")

        z = 1



    else:
        optcial_stack = None



    AMSR2_cropped_stack = open_amsr2(path=AMSR2_path,
                       sensor="AMSR2",
                       overpass="day",
                       subdir_pattern=f"20*",
                       file_pattern="amsr2_l1bt_*.nc",
                       date_pattern=r"_(\d{8})_",
                       time_start=time_start,
                       time_stop=time_stop,
                       resolution = "coarse_resolution",
                       bbox=bbox
                       )


    return  {"SLSTR" : optcial_stack.compute(), "AMSR2" : AMSR2_cropped_stack.compute(),}





# ---------------------------------------
# RAW SATELLITE PROCESSORS
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

    padded_data = clean_pad_data(big_data)

    _dataset = xr.concat(padded_data, dim = "time")
    _dataset = filter_empty_var(_dataset)

    return _dataset.sortby("time")


def open_modis(path,
                    bbox,
                    subdir_pattern: Literal["reflectance","lst"],
                    date_pattern = r"A(\d{7})",
                    time_start="2024-01-01",
                    time_stop="2025-01-01",
                    ):

    folder_modis = os.path.join(path,subdir_pattern,"*.hdf")
    files_modis = glob.glob(folder_modis)
    dates_string=  [re.search(date_pattern, f).group(1) for f in files_modis]
    _dates = pd.to_datetime(dates_string,format =  "%Y%j")
    date_mask  = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))

    files_valid_modis = np.array(files_modis)[date_mask]







