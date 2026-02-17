import xarray as xr
from typing import List, Literal
from pyhdf.SD import SD, SDC
import os
import re
import glob
import pandas as pd
from datacube_utilities import clean_pad_data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()


MODIS_prod_params = {

    "lst": {"qa_band_name": "QC",
            "list_of_flags": [0, 1, 2, 4, 5],
            "variables" : ["LST"]},

    "reflectance" : {"qa_band_name": "1km Reflectance Data State QA",
                     "list_of_flags": [0, 1, 2, 8, 9, 12, 15],
                     "variables" : ['1km Surface Reflectance Band 1', '1km Surface Reflectance Band 5'],
                     }
}


def ndvi_calc(red,nir):
    return ((nir-red)/(nir+red)).rename("NDVI")

def mask_bit_flag(qa_array, bits=None):
    """
    For MYD09 Surface Reflectance see:
        See Table 13: https://modis-land.gsfc.nasa.gov/pdf/MOD09_UserGuide_v1.4.pdf
        For the input array, from MYD09 L2 MODIS surface reflectance data, returns a mask where:
            - Clouds are filtered (Bits set: 0,1,2,8,9)
            - Snow and Ice (Bits set: 12,15)

    For MYD11 LST see:
        See Table 9: https://www.earthdata.nasa.gov/s3fs-public/2025-04/MOD11_User_Guide_V5.pdf?VersionId=DWqcQ5V29aHlBv0O6Ef1_1xNffsdfXqy
        For the input array, from MYD09 L2 MODIS surface reflectance data, returns a mask where:
            - Clouds are filtered (Bits set: 0,1,2,8,9)
            - Snow and Ice (Bits set: 12,15)

    :param array: Input L2 Swath from MODIS QA band!!
    :return: Boolean mask
    """
    if len(bits) == 1:
        mask_array = (qa_array.astype(int) & 2 ** bits) == 2 ** bits
    elif len(bits) > 1:
        combined_mask = sum(1 << b for b in bits)
        mask_array = (qa_array.astype(int)  & combined_mask) > 0
    else:
        mask_array = np.full(shape=qa_array.shape, fill_value = False)

    return mask_array

def pad_array(array_dict, lat, lon):
    array_padded = {}
    for key,array in array_dict.items():

        if array.shape != lat.shape or array.shape != lon.shape:
            array_padded[key] = np.pad(array,
                                [(0,0),(0,1)], #TODO: Revise!
                                mode ="constant", constant_values=np.nan)
        else:
            array_padded[key] = array

    return array_padded

def merge_datasets(ds_NDVI,ds_LST):

    NDVI_masked = ds_NDVI.where(ds_LST > 0).rename("NDVI")
    big_ds = xr.merge([NDVI_masked,ds_LST])[["NDVI","LST"]]
    return big_ds

def xr_from_arrays(data_dict,lat,lon,time, bbox):

    mask = (
            (lat >= bbox[1]) & (lat <= bbox[3]) &
            (lon >= bbox[0]) & (lon <= bbox[2])
    )

    if np.any(mask):
        rows, cols = np.where(mask)
        row_slice = slice(rows.min(), rows.max() + 1)
        col_slice = slice(cols.min(), cols.max() + 1)

        dataset   = xr.Dataset(
            {
                k: (("row", "column"), v[row_slice, col_slice])
                for k, v in data_dict.items()
            },
            coords={
                "lat": (("row", "column"), lat[row_slice, col_slice]),
                "lon": (("row", "column"), lon[row_slice, col_slice]),
            },
        ).assign_coords(time = time)
        print(f"{time} opened")
    else:
        dataset = None
        print(f"{time} has no overlap with bbox")


    return dataset


def open_hdf(path,
             type_of_product: Literal["reflectance", "lst"],
             ):

    data = SD(path)
    var_data_dict = {}

    qa_band_name = MODIS_prod_params[type_of_product]["qa_band_name"]
    qa_array = data.select(qa_band_name)[:].astype(np.uint16)

    flag_bits =  MODIS_prod_params[type_of_product]["list_of_flags"]
    qa_mask = mask_bit_flag(qa_array, flag_bits)

    vars = MODIS_prod_params[type_of_product]["variables"]
    for v in vars:
        v_data = data.select(v)
        _SD_var_array = v_data[:].astype(np.float64)

        SD_var_array = np.where(qa_mask,np.nan,_SD_var_array) # QA masking

        attrs = v_data.attributes()
        scale = attrs["scale_factor"]
        offset = attrs["add_offset"]
        fillvalue = attrs['_FillValue']

        valid_array = np.where(SD_var_array == fillvalue, np.nan, SD_var_array)
        scaled_valid_array = (valid_array + offset ) * scale
        var_data_dict[v] = scaled_valid_array

    lat_var = data.select("Latitude")
    lat_array = lat_var[:].astype(np.float64)
    lon_var = data.select("Longitude")
    lon_array = lon_var[:].astype(np.float64)

    return {"data": var_data_dict, "lat": lat_array, "lon" : lon_array}


def open_modis(path,
               type_of_product: Literal["reflectance", "lst"],
               bbox,
               date_pattern=r"\d{7}\.\d{4}",
               time_start="2024-01-01",
               time_stop="2025-01-01",
               ):

    folder_modis = os.path.join(path, type_of_product, "*.hdf")
    files_modis = glob.glob(folder_modis)

    dates_string = [re.search(date_pattern, f).group(0) for f in files_modis]
    _dates = pd.to_datetime(dates_string, format="%Y%j.%H%M")
    date_mask = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))

    dates_valid_modis = np.array(_dates)[date_mask]
    files_valid_modis = np.array(files_modis)[date_mask]
    MODIS_timeseries = []

    for f, d in zip(files_valid_modis, dates_valid_modis):

        day_data_dict = open_hdf(f, type_of_product= type_of_product)
        day_data_dict["data"] = pad_array(day_data_dict["data"], day_data_dict["lat"], day_data_dict["lon"])
        da_MODIS_day = xr_from_arrays(day_data_dict["data"], day_data_dict["lat"], day_data_dict["lon"],
                                      time=d, bbox=bbox)
        MODIS_timeseries.append(da_MODIS_day)

    padded_data = clean_pad_data(MODIS_timeseries, x="row", y="column")

    MODIS_data = xr.concat(padded_data, dim="time")

    return MODIS_data

if __name__== "__main__":

    myd3_path = "/home/ddkovacs/Downloads/MYD03.A2018001.1945.061.2018002153806.hdf"
    myd9_path =  "/home/ddkovacs/Desktop/modis/midwest/reflectance/MYD09.A2018001.1945.061.2021294135456.hdf"

    myd3_hdf = SD(myd3_path)
    myd9_hdf = SD(myd9_path)

    myd3 = myd3_hdf.datasets()
    myd9 = myd9_hdf.datasets()

    myd3_lat = myd3_hdf.select("Latitude")[:]
    myd9_lat = myd9_hdf.select("Latitude")[:]
