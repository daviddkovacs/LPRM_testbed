import rioxarray as rxr
from osgeo import gdal
import xarray as xr
from typing import List
import rasterio
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
path_lst = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/"
        "daytime_retrieval/LST/MODIS/midwest/lst/MYD11_L2.A2018003.1930.061.2021316030624.hdf")
path_sr = "/home/ddkovacs/Desktop/MYD09.A2018365.2010.061.2021348142533.hdf"

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


SR_var_list = ['1km Surface Reflectance Band 1', '1km Surface Reflectance Band 5']

SR_dict = open_hdf(path_sr, SR_var_list)
SR_dict["data"] = pad_array(SR_dict["data"], SR_dict["lat"], SR_dict["lon"])
ds_SR = xr_from_arrays(SR_dict["data"], SR_dict["lat"], SR_dict["lon"])

LST_dict = open_hdf(path_lst, 'LST')
LST_dict["data"] = pad_array(LST_dict["data"],SR_dict["lat"],SR_dict["lon"])
ds_LST = xr_from_arrays(LST_dict["data"], SR_dict["lat"], SR_dict["lon"])




plt.figure()
ds_LST["LST"].plot.pcolormesh(x= "lon", y= "lat")
plt.show()

plt.figure()
ds_SR["LST"].plot.pcolormesh(x= "lon", y= "lat")
plt.show(block=True)