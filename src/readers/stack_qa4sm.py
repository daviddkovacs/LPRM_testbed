import xarray as xr
import os
import glob
import re
import datetime

path_base = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/04_retrieved/coarse_resolution"
sensor = "AMSR2"
overpass = "day"
year = "2024"
path_overpass = os.path.join(path_base,sensor, overpass,f"{year}**" ,)

fname_pattern = f"{sensor.upper()}_LPRM_VEGC_{overpass}*.nc"
files = glob.glob(os.path.join(path_overpass,fname_pattern))



def _time_add(ds):
    filename = ds.encoding['source']
    date_str = re.search('(\d{4}\d{2}\d{2})', filename).group(1)
    date_obj = datetime.datetime.strptime(date_str,"%Y%m%d")
    _ds = ds.expand_dims({"time":[date_obj]})
    return _ds


data = xr.open_mfdataset(files,
                         combine="by_coords",
                         preprocess=_time_add)


data.to_netcdf(
    f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/validation/{sensor.upper()}_{overpass}_{year}.nc",
    encoding={key:
                  {'zlib': True,'complevel': 5} for key,_ in data.items()}
)
