import os
import glob
from dask.array import block
from config.paths import path_bt
import matplotlib
import numpy as np
from sklearn.linear_model import HuberRegressor
import xarray as xr
import matplotlib.pyplot as plt
from config.paths import path_lprm, path_bt, path_aux
from lprm.retrieval.lprm_v6_1.parameters import get_lprm_parameters_for_frequency
from lprm.satellite_specs import get_specs
from simulator.radiative_transfer_lprm import radiative_transfer
from osgeo import gdal
from utilities.retrieval_helpers import get_coords
from utilities.utils import (
    bbox,
    mpdi,
    extreme_hull_vals,
    find_common_coords,
    get_dates,
    convex_hull,
    pearson_corr,
    save_nc,
)
matplotlib.use("TkAgg")

frequencies={'C1': 6.9, 'C2': 7.3, 'X': 10.7,'KU': 18.7}

start_date = "2024-01-01"
end_date = "2024-12-31"

overpass = "night"
datelist = get_dates(start_date, end_date, freq = "ME")

for d in datelist:
    pattern = f"AMSR2_LPRM_VEGC_{overpass}{d.strftime("%Y%m%d")}_25km_v061.nc"
    bt_filepath = os.path.join(path_lprm, "AMSR2", overpass, d.strftime("%Y%m"), pattern)

    lprm_data = xr.open_dataset(bt_filepath, decode_timedelta=False).assign_coords(lat = get_coords()["coords"]["lat"],
                                                                                   lon = get_coords()["coords"]["lon"],)

    bbox = [
        -139.82872793461695,
        -57.252547698983435,
        173.11658029116143,
        71.11406642771769
      ]
    da = lprm_data.sel(lat=slice(bbox[3], bbox[1]),
                lon=slice(bbox[0], bbox[2]))

    VOD_C1 = da["VOD_C1"]
    VOD_C2 = da["VOD_C2"]
    VOD_X = da["VOD_X"]
    VOD_KU = da["VOD_KU"]

    # means = [np.nanmean(vod_c1),np.nanmean(vod_c2),np.nanmean(vod_x),np.nanmean(vod_ku),]
    res = VOD_KU / VOD_C1

    plt.figure()
    # plt.plot(frequencies.values(),means, "o--")
    # plt.xlim([6,19])
    # plt.ylim([0.4,0.8])
    res.plot(vmin = 0,vmax = 4)
    plt.title("VOD_KU / VOD_C1")
    plt.plot()







