import rioxarray as rxr
from osgeo import gdal
import xarray as xr
from typing import List
import rasterio
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import numpy as np
# path = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/"
#         "daytime_retrieval/LST/MODIS/midwest/lst/MYD11_L2.A2018003.1930.061.2021316030624.hdf")
path_sr = "/home/ddkovacs/Desktop/MYD09.A2018365.2010.061.2021348142533.hdf"

def open_hdf(path, var: List):

    data = SD(path)
    SDvar = data.select(var)
    SD_var_array = SDvar[:].astype(np.float64)
    attrs = SDvar.attributes()
    scale = attrs["scale_factor"]
    offset = attrs["add_offset"]
    fillvalue = attrs['_FillValue']

    valid_array = np.where(SD_var_array == fillvalue,np.nan,SD_var_array)
    scaled_valid_array = (valid_array + offset ) * scale

    lat_var = data.select("Latitude")
    lat_array = lat_var[:].astype(np.float64)

    lon_var = data.select("Longitude")
    lon_array = lon_var[:].astype(np.float64)

    return {"data":scaled_valid_array, "lat": lat_array, "lon" : lon_array}


def pad_array(array): # TODO: maybe needs to edit the shape

    array_padded = np.pad(array,
                        [(0,0),(0,1)],
                        mode ="constant", constant_values=np.nan)
    return array_padded


def xr_from_arrays(data,lat,lon):

    dataset = xr.Dataset(
        {
            "LST": (("row", "column"), data)
        },
        coords={
            "lat": (("row", "column"), np.repeat(np.repeat(lat, 5, axis=0), 5, axis=1)),
            "lon": (("row", "column"), np.repeat(np.repeat(lon, 5, axis=0), 5, axis=1)),
        },
    )

    return dataset


SR_dict = open_hdf(path_sr, '1km Surface Reflectance Band 5')
SR_dict["data"] = pad_array(SR_dict["data"])
ds_SR = xr_from_arrays(SR_dict["data"], SR_dict["lat"], SR_dict["lon"])




plt.figure()
ds_LST["LST"].plot.pcolormesh(x= "lon", y= "lat")
plt.show(block=True)
