import xarray as xr
import os
import numpy as np
import glob
file = "AMSR2_day_2024"
path = f"/home/david/mounted_climers01/home/ddkovacs/Desktop/daytime_validation/{file}.nc"

ds = xr.open_dataset(path,decode_timedelta=False)

ds = ds.rename({'LAT': 'lat_matrix', 'LON': 'lon_matrix'})

# 2. Define proper 1D coordinate arrays for the grid dimensions (adjust ranges as needed for your data)
lat_1d = np.linspace(-90, 90, 720)
lon_1d = np.linspace(-180, 180, 1440)

# 3. Assign them as valid 1D coordinates to satisfy NetCDF validation
ds = ds.assign_coords(
    lat=('lat', lat_1d),
    lon=('lon', lon_1d)
)
compression_settings = {"zlib": True, "complevel": 5}

ds.to_netcdf(
    f"/home/david/mounted_climers01/home/ddkovacs/Desktop/daytime_validation/{file}_comp.nc",
    encoding={var : compression_settings for var in ds}
)