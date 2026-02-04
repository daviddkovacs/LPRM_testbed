import glob
import re
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
from config.paths import  SLSTR_path
import pandas as pd
from datetime import datetime

def threshold_ndvi(lst, ndvi, ndvi_thres=0.3):
    """
    Simple thresholding of Soil-Veg to get different temps.
    """
    veg_temp = xr.where(ndvi >ndvi_thres, lst, np.nan)
    soil_temp = xr.where(ndvi <ndvi_thres, lst, np.nan)

    return soil_temp, veg_temp


def crop2roi(ds,bbox):
    """
    Cropping to bbox. Handles S3 projection (lat, lon) for each coord
    :param ds:
    :param bbox:
    :return:
    """
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


def subset_statistics(array):

    _array = filternan(array)
    stat_dict = {}
    if np.any(~np.isnan(_array)):
        stat_dict["mean"] = np.nanmean(_array).item()
        stat_dict["std"] = np.nanstd(_array).item()
    else:
        stat_dict["mean"] = np.nan
        stat_dict["std"] = np.nan
    return _array, stat_dict

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
                                chunks = "auto",
                                decode_timedelta = False).assign_coords(time = _dates[date_mask])

    print(f"Loading dataset finished (AMSR2)")

    return dataset


def open_sltsr(path,
               subdir_pattern,
               date_pattern,
               variable_file,
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
                                chunks = "auto",
                                decode_timedelta=False,
                                ).assign_coords(time = dates_dt)

    if georeference_file: # L1 and L2 SLSTR data isnt gridded. lat, lon from external file!

        coord_path = os.path.join(path,subdir_pattern,georeference_file)
        geo_files = glob.glob(coord_path)

        geo = xr.open_mfdataset(geo_files,
                                preprocess =clip_swath,
                                combine="nested",
                                join="outer",
                                concat_dim="time",
                                chunks="auto",
                                decode_timedelta=False,
                                ).assign_coords(time = dates_dt)

        dataset = dataset.assign_coords(
            lat=(("time", "rows", "columns"), geo.latitude_in.data),
            lon=(("time", "rows", "columns"), geo.longitude_in.data)
        )

        print(f"Loading dataset finished ({variable_file})")

    return dataset


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


def preprocess_slstr(NDVI,LST):
    """
    Merge LST and NDVI, then Cloud, snow filtering and clearing possibly empty NDVI observations
    """
    _SLSTR = xr.merge([NDVI,LST])[["LST","NDVI"]]
    _SLSTR = cloud_filtering(_SLSTR) # Mask clouds (strict)
    _SLSTR = snow_filtering(_SLSTR) # Mask clouds (strict)

    return filter_empty_var(_SLSTR, "NDVI") # Filter empty NDVI obs


def get_edges(centers):
    """
    Calculate the spacing between pixels, to properly handle np.digitize. Otherwise offset.
    """
    res = np.abs(np.diff(centers)[0])

    edges = np.append(np.sort(centers) - res / 2, np.sort(centers)[-1] + res / 2)
    return np.sort(edges)


def binning_smaller_pixels(slstr_da,amsr2_da):

    lat_edges = get_edges(amsr2_da.lat.values)
    lon_edges = get_edges(amsr2_da.lon.values)

    iterables = {}

    iterables["lats"] = np.digitize(slstr_da.lat.values, lat_edges) - 1
    iterables["lons"] = np.digitize(slstr_da.lon.values, lon_edges) - 1

    return iterables


def slstr_pixels_in_amsr2(slstr_da,
                          bin_dict,
                          target_lat_bin,
                          target_lon_bin):

    mask = (bin_dict["lats"]  == target_lat_bin) & (bin_dict["lons"]  == target_lon_bin)
    pixels_within = slstr_da.where(xr.DataArray(mask, coords=slstr_da.coords), drop=True)

    return pixels_within


def compare_temperatures(soil_temp, veg_temp, TSURF, ):
    """
    Gets the underlying SLSTR pixels for every AMSR2 Ka-LST pixel. Then calculates the mean and std for these, and plots
    """
    try:
        veg_mean_list = []
        veg_std_list = []

        soil_mean_list = []
        soil_std_list = []
        TSURF_list = []

        bin_dict = binning_smaller_pixels(soil_temp,
                                          TSURF)  # instead of soil_temp, any shoudl be good thats a SLSTR obs

        for targetlat in range(0, bin_dict["lats"].max()):
            for targetlon in range(0, bin_dict["lons"].max()):

                soil_subset = slstr_pixels_in_amsr2(soil_temp,
                                                    bin_dict,
                                                    targetlat,
                                                    targetlon)

                veg_subset = slstr_pixels_in_amsr2(veg_temp,
                                                   bin_dict,
                                                   targetlat,
                                                   targetlon)

                soil_mean_list.append(subset_statistics(soil_subset)[1]["mean"])
                soil_std_list.append(subset_statistics(soil_subset)[1]["std"])

                veg_mean_list.append(subset_statistics(veg_subset)[1]["mean"])
                veg_std_list.append(subset_statistics(veg_subset)[1]["std"])

                TSURF_subset = TSURF.isel(lat=targetlat, lon=targetlon)
                TSURF_list.append(TSURF_subset.values.item())

        df =  pd.DataFrame({"veg_mean": veg_mean_list,
                                 "veg_std": veg_std_list,
                                 "soil_mean": soil_mean_list,
                                 "soil_std": soil_std_list,
                                 "tsurf_ka": TSURF_list,
                                 })
        df_sorted = df.sort_values(by="tsurf_ka")

    except Exception as e:
        print(e)
        breakpoint()

    return df_sorted
