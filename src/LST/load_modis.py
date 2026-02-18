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
from config.paths import MODIS_geo_path, MODIS_geo_path_local
import datetime

MODIS_prod_params = {

    "lst": {"qa_band_name": "QC",
            # "list_of_flags": [0, 1, 2, 4, 5], #TODO: REVISE!!!!!!!!
            "list_of_flags": [0, ],
            "variables" : ["LST"]},

    "reflectance" : {"qa_band_name": "1km Reflectance Data State QA",
                     # "list_of_flags": [0, 1, 2, 8, 9, 12, 15],
                     "list_of_flags": [0, ],
                     "variables" : ['1km Surface Reflectance Band 1', '1km Surface Reflectance Band 5'],
                     }
}


def ndvi_calc(red,nir):
    return ((nir-red)/(nir+red)).to_dataset(name="NDVI")

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
        bit = bits[0]
        mask_array = (qa_array.astype(int) & 2 ** bit) == 2 ** bit
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
    """
    Selects coords lying within the bbox.
    :param data_dict: dictionary having data[var] lat and lon
    :param lat: lat array
    :param lon: lon array
    :param time: timestamp
    :param bbox: bbox as list
    :return: xr.DataArray()
    """
    mask = (
            (lat >= bbox[1]) & (lat <= bbox[3]) &
            (lon >= bbox[0]) & (lon <= bbox[2])
    )

    if np.any(mask):
        rows, cols = np.where(mask)
        row_slice = slice(rows.min(), rows.max() + 1)
        col_slice = slice(cols.min(), cols.max() + 1)
        print(f"{time} opened")
    else:
        # Create empty slices to generate arrays of shape (0, 0)
        row_slice = slice(0, 0)
        col_slice = slice(0, 0)
        print(f"{time} has no overlap with bbox, returning empty Dataset")

    dataset = xr.Dataset(
        {
            k: (("row", "column"), v[row_slice, col_slice])
            for k, v in data_dict.items()
        },
        coords={
            "lat": (("row", "column"), lat[row_slice, col_slice]),
            "lon": (("row", "column"), lon[row_slice, col_slice]),
        },
    ).assign_coords(time=time)

    return dataset

def apply_attributes(array,attrs):
    """
    Function to apply, scaling, offset and NaN values from HDF-EOS format. note, this is taken care of automatically
    in xarray
    :param array: array to manipulate
    :param attrs: attributes usually obtained by: SD(path).select(var).attributes()
    :return: scaled_valid_array
    """
    scale = attrs["scale_factor"] if "scale_factor" in attrs.keys() else 1
    offset = attrs["add_offset"] if "add_offset" in attrs.keys() else 0
    fillvalue = attrs['_FillValue']

    valid_array = np.where(array == fillvalue, np.nan, array)
    scaled_valid_array = (valid_array + offset) * scale

    return scaled_valid_array

def geolocation_file(datestring: str = None,
                     path = MODIS_geo_path_local,
                     ):
    """
    MODIS LST does NOT have pixel-wise georeferencing, because they wanted to save memory. Thus it is supplied by the
    MYD03 Geolocation product.
    This function accepts the usual datestring from the LST filename, and searches for it in the geolocation file list.
    :param datestring: MODIS file pattern referring for date.hour
    :param path: Path where MYD03 geolocation files are stored
    :return:
    """

    geo_files = glob.glob(os.path.join(path,"MYD03*.hdf"))
    corresponding_geo_file_index = next((i for i, f in enumerate(geo_files) if datestring in f), None)
    corresponding_geo_file = geo_files[corresponding_geo_file_index]

    geo_data = SD(corresponding_geo_file)
    lat = geo_data.select("Latitude")
    lon = geo_data.select("Longitude")
    _lat_array = lat[:].astype(np.float64)
    _lon_array = lon[:].astype(np.float64)

    lat_array = apply_attributes(_lat_array, lat.attributes())
    lon_array = apply_attributes(_lon_array, lon.attributes())

    return lat_array, lon_array


def open_hdf(path,
             type_of_product: Literal["reflectance", "lst"],
             datestring: str = None,  # Format: YYYYDOY.HHMM
             geo_path = MODIS_geo_path
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
        var_attrs = v_data.attributes()
        var_data_dict[v] = apply_attributes(SD_var_array, var_attrs)

    if type_of_product == "lst":
        # MYD11 LST data has no pixel-wise geolocation data! Needs to be read from MYD03
        lat_array, lon_array = geolocation_file(datestring=datestring, path=geo_path)

    elif type_of_product == "reflectance":
        # MYD09 Does have geolocation within the same file.
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
               geo_path = MODIS_geo_path
               ):

    folder_modis = os.path.join(path, type_of_product, "*.hdf")
    files_modis = glob.glob(folder_modis)

    dates_string = [re.search(date_pattern, f).group(0) for f in files_modis]
    _dates = pd.to_datetime(dates_string, format="%Y%j.%H%M")
    date_mask = (pd.to_datetime(time_start) < _dates) & (_dates < pd.to_datetime(time_stop))

    dates_string_valid =  np.array(dates_string)[date_mask]
    dates_valid_modis = np.array(_dates)[date_mask]
    files_valid_modis = np.array(files_modis)[date_mask]
    MODIS_timeseries = []

    for file, date, datestring in zip(files_valid_modis, dates_valid_modis, dates_string_valid):

        day_data_dict = open_hdf(file, type_of_product= type_of_product,datestring= datestring, geo_path = geo_path)
        day_data_dict["data"] = pad_array(day_data_dict["data"], day_data_dict["lat"], day_data_dict["lon"])
        da_MODIS_day = xr_from_arrays(day_data_dict["data"], day_data_dict["lat"], day_data_dict["lon"],
                                      time=date, bbox=bbox)
        MODIS_timeseries.append(da_MODIS_day)

    padded_data = clean_pad_data(MODIS_timeseries, x="row", y="column")

    MODIS_data = xr.concat(padded_data, dim="time")

    return MODIS_data.sortby("time")

if __name__== "__main__":

    myd3_path = "/home/ddkovacs/Downloads/MYD03.A2018001.1945.061.2018002153806.hdf"
    myd9_path =  "/home/ddkovacs/Desktop/modis/midwest/reflectance/MYD09.A2018001.1945.061.2021294135456.hdf"

    myd3_hdf = SD(myd3_path)
    myd9_hdf = SD(myd9_path)

    myd3 = myd3_hdf.datasets()
    myd9 = myd9_hdf.datasets()

    myd3_lat = myd3_hdf.select("Latitude")[:]
    myd9_lat = myd9_hdf.select("Latitude")[:]
