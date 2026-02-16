import xarray as xr
from typing import List, Literal
from pyhdf.SD import SD, SDC
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

path_lst = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/"
        "daytime_retrieval/LST/MODIS/midwest/lst/MYD11_L2.A2018003.1930.061.2021316030624.hdf")
path_sr = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/LST/MODIS/midwest/"


bbox = []

def open_modis(path,
                    bbox,
                    subdir_pattern: Literal["reflectance","lst"],
                    date_pattern = r"\d{7}\.\d{4}",
                    time_start="2024-01-01",
                    time_stop="2025-01-01",
                    ):

    folder_modis = os.path.join(path,subdir_pattern,"*.hdf")
    files_modis = glob.glob(folder_modis)

    dates_string=  [re.search(date_pattern, f).group(0) for f in files_modis]
    _dates = pd.to_datetime(dates_string,format =  "%Y%j.%H%M")
    date_mask  = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))

    files_valid_modis = np.array(files_modis)[date_mask]

open_modis(path_sr,bbox,subdir_pattern="reflectance", time_stop="2019-01-01", time_start="2018-01-01")



def open_hdf(path, var: List):
    data = SD(path)
    var_data_dict = {}

    for v in var:
        v_data = data.select(v)
        SD_var_array = v_data[:].astype(np.float64)
        attrs = v_data.attributes()
        scale = attrs["scale_factor"]
        offset = attrs["add_offset"]
        fillvalue = attrs['_FillValue']

        valid_array = np.where(SD_var_array == fillvalue,np.nan,SD_var_array)
        scaled_valid_array = (valid_array + offset ) * scale
        var_data_dict[v] = scaled_valid_array

    lat_var = data.select("Latitude")
    lat_array = lat_var[:].astype(np.float64)
    lon_var = data.select("Longitude")
    lon_array = lon_var[:].astype(np.float64)

    return {"data": var_data_dict, "lat": lat_array, "lon" : lon_array}


def pad_array(array_dict, lat, lon):
    array_padded = {}
    for key,array in array_dict.items():

        if array.shape != lat.shape or array.shape != lon.shape:
            array_padded[key] = np.pad(array,
                                [(0,0),(0,1)],
                                mode ="constant", constant_values=np.nan)
        else:
            array_padded[key] = array

    return array_padded


def xr_from_arrays(data_dict,lat,lon):

    dataset = xr.Dataset(
        {
            k: (("row", "column"), v) for k, v in data_dict.items()
         },
        coords={
            "lat": (("row", "column"), lat),
            "lon": (("row", "column"), lon),
        },
    )

    return dataset

def ndvi_calc(red,nir):
    return ((nir-red)/(nir+red)).rename("NDVI")


def merge_datasets(ds_NDVI,ds_LST):

    NDVI_masked = ds_NDVI.where(ds_LST > 0).rename("NDVI")
    big_ds = xr.merge([NDVI_masked,ds_LST])[["NDVI","LST"]]
    return big_ds


SR_var_list = ['1km Surface Reflectance Band 1', '1km Surface Reflectance Band 5']

SR_dict = open_hdf(path_sr, SR_var_list)
SR_dict["data"] = pad_array(SR_dict["data"], SR_dict["lat"], SR_dict["lon"])
ds_SR = xr_from_arrays(SR_dict["data"], SR_dict["lat"], SR_dict["lon"])
ndvi = ndvi_calc(ds_SR['1km Surface Reflectance Band 1'], ds_SR['1km Surface Reflectance Band 5'])


LST_dict = open_hdf(path_lst, ['LST'])
LST_dict["data"] = pad_array(LST_dict["data"],SR_dict["lat"],SR_dict["lon"])
ds_LST = xr_from_arrays(LST_dict["data"], SR_dict["lat"], SR_dict["lon"])

big_ds = merge_datasets(ndvi,ds_LST["LST"])



x =1
plt.figure()
big_ds['NDVI'].plot.pcolormesh(x= "lon", y= "lat", vmin =0, vmax =1)
plt.show(block=True)

plt.figure()
big_ds['LST'].plot.pcolormesh(x= "lon", y= "lat")
plt.show(block=True)