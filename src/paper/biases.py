import xarray as xr
import glob
import os
import matplotlib.pyplot as plt

path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/04_retrieved/coarse_resolution"
sensor = "AMSR2"
month = "2024*"

day_path = os.path.join(path,sensor,"day",month,"*.nc")
night_path = os.path.join(path,sensor,"night",month,"*.nc")

day_files  = glob.glob(day_path)
night_files  = glob.glob(night_path)

day_datasets = xr.open_mfdataset(
    day_files, combine="nested", concat_dim="time", decode_timedelta=False
)

night_datasets = xr.open_mfdataset(
    night_files, combine="nested", concat_dim="time", decode_timedelta=False
)

##
freq = "X"

vod_day = day_datasets[f"VOD_{freq}"]
vod_night = night_datasets[f"VOD_{freq}"]

bias = (vod_day.mean(dim = "time") -vod_night.mean(dim = "time") ).compute()


##
plt.figure()
bias.plot(figsize = (20,10),
          vmin = -0.1,vmax =0.1,
          cmap = "coolwarm")
plt.title(f"Bias Day-Night {freq} VOD 2024 ")
plt.show()