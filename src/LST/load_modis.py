import xarray as xr
from typing import Literal
from pyhdf.SD import SD
import os
import re
import glob
import pandas as pd
from datacube_utilities import clean_pad_data
import numpy as np
from config.paths import MODIS_geo_path, MODIS_geo_path_local
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

MODIS_prod_params = {
    "reflectance":
        {"qa_band_name": "1km Reflectance Data State QA",
         "list_of_flags": [
             {"bits": [0, 1], "vals": [1,2]},  # Cloud State
             {"bits": [2], "vals": [1]},  # Cloud shadow
             {"bits": [3,4,5], "vals": [3,]},  # land/water flag
             {"bits": [8,9], "vals": [1,2,3]},  # cirrus
             {"bits": [12], "vals": [1]},  # snow/ice
         ],
         "variables":
             ['1km Surface Reflectance Band 1', '1km Surface Reflectance Band 5'],
         },

    "lst":
        {
            "qa_band_name": "QC",
            "list_of_flags": [
                {"bits": [0, 1], "vals": [2]},  # Mandatory QA flags
                {"bits": [2, 3], "vals": [3]},  # Data quality flag
                {"bits": [4, 5], "vals": [1,3]},  # Cloud flag
            ],
            "variables" : ["LST"]
        },
}


def ndvi_calc(red,nir):
    return ((nir-red)/(nir+red)).to_dataset(name="NDVI")


def mask_bit_flag(qa_array, bits, to_be_masked_values):
    """
    qa_array: The raw QA band (uint16)
    bits: list of bit positions (e.g., [0, 1])
    to_be_masked_values: list of decimal states to keep (e.g., [1, 2])
    """
    qa_int = qa_array.astype(np.uint16)
    bit_mask = sum(1 << b for b in bits)

    shift_amount = min(bits)
    extracted_values = (qa_int & bit_mask) >> shift_amount

    return np.isin(extracted_values, to_be_masked_values)


def pad_array(array_dict, lat, lon):
    """
    Sometimes the MODIS L2 Swath needs to be padded, otherwise the datacube will not construct.
    Unfortunately, this is ugly, but at least it works.
    :param array_dict: MODIS array to be padded (if needed)
    :param lat: Latitude array
    :param lon: Longitude array
    :return: If needed a padded array to the shape of Lat and Lon
    """
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


def xr_from_arrays(data_dict,lat,lon,time, bbox, buffer = 0.1):
    """
    Selects coords lying within the bbox.
    buffer: in order to allow for cropping to AMSR2 extent, we first assign a larger ROI than the bbox
    """
    mask = (
            (lat >= bbox[1]-buffer) & (lat <= bbox[3]+buffer) &
            (lon >= bbox[0]-buffer) & (lon <= bbox[2]+buffer)
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

    return (lat_array), (lon_array)


def open_hdf(path,
             type_of_product: Literal["reflectance", "lst"],
             datestring: str = None,  # Format: YYYYDOY.HHMM
             geo_path = MODIS_geo_path
             ):

    data = SD(path)
    var_data_dict = {}

    qa_band_name = MODIS_prod_params[type_of_product]["qa_band_name"]
    qa_array = data.select(qa_band_name)[:].astype(np.uint16)

    # We will add the invalid pixels to the final mask
    final_qa_mask = np.zeros(qa_array.shape, dtype=bool)

    for filter_set in MODIS_prod_params[type_of_product]["list_of_flags"]:
        # This finds pixels that shold be flagged. They are grouped.
        current_mask = mask_bit_flag(qa_array, filter_set["bits"], filter_set["vals"])
        final_qa_mask = final_qa_mask | current_mask

    vars = MODIS_prod_params[type_of_product]["variables"]
    for v in vars:

        v_data = data.select(v)
        _SD_var_array = v_data[:].astype(np.float64)
        SD_var_array = np.where(final_qa_mask,np.nan,_SD_var_array) # QA masking
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


    # plt.figure()
    # mesh = plt.pcolormesh(lon_array, lat_array, SD_var_array,
    #                       cmap='viridis',
    #                       shading='auto')
    # plt.draw()
    # x = 1
    # plt.show()


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
    files_modis = sorted(glob.glob(folder_modis))

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