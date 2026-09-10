import glob
import os
import numpy as np
import xarray as xr
variable = "X"
file = "AMSR2_day_2024"
path = f"/home/ddkovacs/Desktop/daytime_validation/{file}.nc"

ds = xr.open_dataset(path, decode_timedelta=False)

# Completely remove the old matrix variables/coordinates if they exist
ds = ds.drop_vars(["LAT", "LON", "lat_matrix", "lon_matrix"], errors="ignore")

# Define proper 1D coordinate arrays for the grid dimensions
lat_1d = np.linspace(-90, 90, 720)
lon_1d = np.linspace(-180, 180, 1440)

# Assign 1D coordinates with long_name attributes included directly
ds = ds.assign_coords(
    lat=("lat", lat_1d, {"long_name": "latitude"}),
    lon=("lon", lon_1d, {"long_name": "longitude"}),
)
ds = ds[f"SM_{variable}"]
compression_settings = {"zlib": True, "complevel": 5}

ds.to_netcdf(
    f"/home/ddkovacs/Desktop/daytime_validation/{file}_{variable}_comp_nano.nc",
    encoding={f"SM_{variable}": compression_settings},
)
