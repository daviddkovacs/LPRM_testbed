import rioxarray as rxr
from osgeo import gdal
import xarray as xr
import rasterio
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import numpy as np
path = ("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/"
        "daytime_retrieval/LST/MODIS/midwest/lst/MYD11_L2.A2018003.1930.061.2021316030624.hdf")


def open_hdf(path, var):

    data = SD(path)
    LST_var = data.select(var)
    _LST_array = LST_var[:].astype(np.float64)
    LST_attrs = LST_var.attributes()
    LST_scale = LST_attrs["scale_factor"]
    LST_offset = LST_attrs["add_offset"]
    LST_fillvalue = LST_attrs['_FillValue']

    LST_array = np.where(_LST_array == LST_fillvalue,np.nan,_LST_array)
    LST_array_calibrated = (LST_array + LST_offset ) * LST_scale

    lat_var = data.select("Latitude")
    lat_array = lat_var[:].astype(np.float64)

    lon_var = data.select("Longitude")
    lon_array = lon_var[:].astype(np.float64)

    return {"data":LST_array_calibrated, "lat": lat_array, "lon" : lon_array}


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


data_dict = open_hdf(path, "LST")
data_dict["data"] = pad_array(data_dict["data"])
ds_LST = xr_from_arrays(data_dict["data"], data_dict["lat"], data_dict["lon"])

plt.figure()
ds_LST["LST"].plot.pcolormesh(x= "lon", y= "lat")
plt.show(block=True)
