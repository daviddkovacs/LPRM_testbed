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




def ndvi_calc(red,nir):
    return ((nir-red)/(nir+red)).rename("NDVI")

def mask_bit_flag(array, bit):
    """
    For the input array, from MYD09 L2 MODIS surface reflectance data, returns a mask where:
        - Clouds are filtered (Bits set: 0,1,2,8,9)
        - Snow and Ice (Bits set: 12,15)

    See Table 13: https://modis-land.gsfc.nasa.gov/pdf/MOD09_UserGuide_v1.4.pdf

    :param array: Input L2 Swath from MODIS
    :return: Boolean mask
    """
    mask = (array.astype(int) & 2 ** bit) == 2 ** bit
    return mask

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


def open_hdf(path, var: List):
    data = SD(path)
    var_data_dict = {}

    qa_array = data.select("1km Reflectance Data State QA")[:].astype(np.float64)
    qa_mask = (mask_bit_flag(qa_array,0) |
               mask_bit_flag(qa_array,1) |
               mask_bit_flag(qa_array,2) |
               mask_bit_flag(qa_array,8) |
               mask_bit_flag(qa_array,9) |
               mask_bit_flag(qa_array,12) |
               mask_bit_flag(qa_array,15)
               )

    for v in var:
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


def open_modis_timeseries(path,
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

        if type_of_product.lower() == "reflectance":
            variables = ['1km Surface Reflectance Band 1', '1km Surface Reflectance Band 5']
        elif type_of_product.lower() == "lst":
            variables = ["LST"]

        day_data_dict = open_hdf(f, variables)

        day_data_dict["data"] = pad_array(day_data_dict["data"], day_data_dict["lat"], day_data_dict["lon"])
        da_MODIS_day = xr_from_arrays(day_data_dict["data"], day_data_dict["lat"], day_data_dict["lon"],
                                      time=d, bbox=bbox)
        MODIS_timeseries.append(da_MODIS_day)

    padded_data = clean_pad_data(MODIS_timeseries, x="row", y="column")

    MODIS_data = xr.concat(padded_data, dim="time")

    return MODIS_data

if __name__== "__main__":

    # path_lst = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/"
    #             "daytime_retrieval/LST/MODIS/midwest/lst/MYD11_L2.A2018003.1930.061.2021316030624.hdf")
    path_sr = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/LST/MODIS/midwest/"

    bbox = [
        -104.47526565142171,
        36.88112420551842,
        -103.97963676129571,
        37.16747407362031
    ]

    ds  = open_modis_timeseries(path_sr, type_of_product="reflectance", bbox=bbox, time_start="2018-01-01",
                          time_stop="2018-02-01")
    ds_ndvi = ndvi_calc(ds["1km Surface Reflectance Band 1"],
                        ds["1km Surface Reflectance Band 5"],)


    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.TIME
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
