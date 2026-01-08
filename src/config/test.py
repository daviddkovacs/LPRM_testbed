import os
import glob
from config.paths import path_bt, path_lprm
import pandas as pd

import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import xarray as xr
from lprm.retrieval.lprm_v6_1.parameters import get_lprm_parameters_for_frequency
from lprm.satellite_specs import get_specs
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from shapely.geometry import LineString,  Point
from lprm.retrieval.lprm_v6_1.run_lprmv6 import load_band_from_ds

year = "2024"
sat_band = "C1"
sat_sensor = "amsr2"
def cut_2_roi(ds,bbox= [-135.31, 22.1, -65.26,51.27]):

    fname = os.path.basename(ds.encoding["source"])
    if "time" not in ds.dims:
        date_str = fname.split('night')[1][:8]
        time_stamp = pd.to_datetime(date_str)
    ds = ds.rename({var: var.upper() for var in ds.dims})
    for coord in ["LAT", "LON"]:
        if coord in ds.data_vars:
            ds = ds.set_coords(coord)
    ds = ds.set_index(LAT="LAT",
                 LON = "LON",
                      )
    ds = ds.sortby(["LAT", "LON"])

    if "LPRM" in fname:
        ds = ds.expand_dims(TIME=[time_stamp])

    _ds = ds.sel(
        LON=slice(bbox[0], bbox[2]),
        LAT=slice(bbox[1], bbox[3])
    )

    return _ds

bt_path = os.path.join(path_bt,"day",f"{year}*", f"*day_{year}*.nc")
bt_files = glob.glob(bt_path)

lprm_path = os.path.join(path_lprm,"night",f"{year}*", f"*night{year}*.nc")
lprm_files = glob.glob(lprm_path)

day_bt = xr.open_mfdataset(bt_files, preprocess=cut_2_roi, parallel=True)
night_lprm = xr.open_mfdataset(lprm_files, preprocess=cut_2_roi, parallel=True)

vod = night_lprm["VOD_KU"]
teff = night_lprm["TSURF"]
# plt.figure()
# plt.hist(vod, bins=1000)
# plt.show(block=True)

vod_max = np.nanmean(vod) + 2 * np.nanstd(vod)
vod = np.where(vod < vod_max, vod, np.nan)
fvc = vod / vod_max

specs = get_specs(sat_sensor.upper())
params = get_lprm_parameters_for_frequency(sat_band, specs.incidence_angle)
freq = get_specs(sat_sensor.upper()).frequencies[sat_band.upper()]

def tiff_df(path,lista = (-180,-90,90,180),target_res = "25"):

    path_BLD = os.path.join(path,f"auxiliary_data_BLD_{target_res}km")
    path_SND = os.path.join(path,f"auxiliary_data_SND_{target_res}km")
    path_CLY = os.path.join(path,f"auxiliary_data_CLY_{target_res}km")

    BLD = get_aux(path_BLD, "BLD")
    SND = get_aux(path_SND, "SND")
    CLY = get_aux(path_CLY, "CLY")

    _common_df = pd.merge(BLD,SND,
                         how='inner', on = ["LON","LAT"],)
    common_df = pd.merge(_common_df,CLY,
                         how='inner', on = ["LON","LAT"],)
    common_df = common_df.set_index(["LAT","LON"])
    subset_panda = bbox(common_df,lista)

    return subset_panda

for date in day_bt.TIME:
    bt_map = day_bt.sel(TIME=date)
    sm, vod = par100.run_band(
        bt_map["BT_V"].values.astype('double'),
        bt_map["BT_H"].values.astype('double'),
        bt_map["TSURF"].values.astype('double'),
        bt_map["SND"].values.astype('double'),
        bt_map["CLY"].values.astype('double'),
        bt_map["BLD"].values.astype('double'),
        params.Q,
        params.w,
        params.opt_atm,
        specs.incidence_angle[0],
        params.h1,
        params.h2,
        params.vod_Av,
        params.vod_Bv,
        float(freq),
        params.temp_freeze,
        False,
        None,
        T_soil=T_soil_test if T_soil_test else merged_geo["T_soil_hull"].values.astype('double'),
        T_canopy=T_canopy_test if T_canopy_test else merged_geo["T_canopy_hull"].values.astype('double'),
    )
