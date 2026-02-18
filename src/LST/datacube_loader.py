import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
from LST.datacube_utilities import crop2roi, get_edges
from LST.load_amsr2 import open_amsr2
from LST.load_slstr import open_sltsr
from LST.load_modis import open_modis,ndvi_calc
from LST.plot_functions import plot_modis_comparison
matplotlib.use("TkAgg")

import os
from typing import Literal, List
from config.paths import S3_SLSTR_path, path_bt, MODIS_path, MODIS_geo_path, MODIS_path_local, MODIS_geo_path_local


# ---------------------------------------
# DATACUBE PROCESSORS
def match_OPTI_to_AMSR2_date(OPTI, AMSR2, date):
    """
    Select the closest date to SLSTR, and thus select this date to access AMSR2
    """
    # SLSTR["time"] = SLSTR.time.sortby("time")
    OPTI = OPTI.drop_duplicates(dim="time")
    OPTI_obs = OPTI.sel(time=date, method="nearest")

    # We select OPTI's observation to get AMSR2. the frequency of obs for AMSR2 is higher.
    AMSR2_obs = AMSR2.sortby('time').sel(time=OPTI_obs.time.dt.floor("d"), method="nearest")

    return OPTI_obs, AMSR2_obs


def spatial_subset_dc(OPTI, AMSR2,  bbox):
    """
    SLSTR is cut to the full spatial extent of AMSR2.
    Both AMSR2 and SLSTR cropped to bbox
    """
    AMSR2_roi = crop2roi(AMSR2, bbox)
    res = AMSR2_roi.attrs["resolution"]

    AMSR2_bbox = [get_edges(AMSR2_roi.lon.values, res).min(),
                  get_edges(AMSR2_roi.lat.values, res).min(),
                  get_edges(AMSR2_roi.lon.values, res).max(),
                  get_edges(AMSR2_roi.lat.values, res).max()]

    OPTI_roi = crop2roi(OPTI, AMSR2_bbox)

    return  OPTI_roi, AMSR2_roi


def OPTICAL_datacube(region: Literal["sahel", "siberia", "midwest", "ceu"],
                         bbox: List[float],
                         sensor: Literal["MODIS", "SLSTR"],
                         time_start = "2024-01-01",
                         time_stop = "2025-01-01",
                         ):
    """
    Main function to obtain SLSTR and AMSR2 observations, cut to the ROI.
    :param date: Date
    :param bbox: Bound box (lonmin, latmin, lonmax, latmax)
    :param optical_path: Path where SLSTR data is stored. Accepts "SL_2_LST*.SEN3" unpacked folders.
    :param region: Region of SLSTR. Currently downloaded: Sahel, Siberia and US Midwest
    :return: dictionary with SLSTR and AMSR2 datacubes.
    """


    if sensor.upper() == "SLSTR":  # DEPRECATED!
        SLSTR_path_region = os.path.join(S3_SLSTR_path, region)

        SLSTR_stack = open_sltsr(SLSTR_path_region,
                                   time_start = time_start,
                                   time_stop = time_stop,
                                   bbox=bbox
                                   ) # Will not work, currently will need to be adjusted!!!

    elif sensor.upper() == "MODIS":
        MODIS_path_region = os.path.join(MODIS_path_local, region)

        MODIS_reflectance = open_modis(MODIS_path_region,
                                   bbox=bbox,
                                   type_of_product="reflectance",
                                   time_start=time_start,
                                   time_stop=time_stop,
                                       geo_path = MODIS_geo_path_local)

        MODIS_NDVI = ndvi_calc(MODIS_reflectance["1km Surface Reflectance Band 1"],
                               MODIS_reflectance["1km Surface Reflectance Band 5"])

        MODIS_LST = open_modis(MODIS_path_region,
                                   bbox=bbox,
                                   type_of_product="lst",
                                   time_start=time_start,
                                   time_stop=time_stop)

        plotdate = "2018-01-14T16:00"
        plot_modis_comparison(MODIS_NDVI["NDVI"], MODIS_LST["LST"], ndvi_time=plotdate,
                              lst_time=plotdate)

        return MODIS_NDVI, MODIS_LST


def MICROWAVE_datacube(
        bbox: List[float],
        path=path_bt,
        sensor="AMSR2",
        overpass:Literal["day","night","daynight"] = "day",
        time_start="2024-01-01",
        time_stop="2025-01-01",
):
    if overpass == "day" or overpass == "night":
        AMSR2_stack = open_amsr2(path=path,
                                         sensor=sensor,
                                         overpass=overpass,
                                         subdir_pattern=f"20*",
                                         file_pattern="amsr2_l1bt_*.nc",
                                         date_pattern=r"_(\d{8})_",
                                         time_start=time_start,
                                         time_stop=time_stop,
                                         resolution = "coarse_resolution",
                                         bbox=bbox
                                         )
    elif overpass == "daynight":
        day_amsr2 = open_amsr2(path=path,
                                         sensor=sensor,
                                         overpass="day",
                                         subdir_pattern=f"20*",
                                         file_pattern="amsr2_l1bt_*.nc",
                                         date_pattern=r"_(\d{8})_",
                                         time_start=time_start,
                                         time_stop=time_stop,
                                         resolution = "coarse_resolution",
                                         bbox=bbox
                                         )

        night_amsr2 = open_amsr2(path=path,
                                         sensor=sensor,
                                         overpass="night",
                                         subdir_pattern=f"20*",
                                         file_pattern="amsr2_l1bt_*.nc",
                                         date_pattern=r"_(\d{8})_",
                                         time_start=time_start,
                                         time_stop=time_stop,
                                         resolution = "coarse_resolution",
                                         bbox=bbox
                                         )
        AMSR2_stack = xr.concat([day_amsr2, night_amsr2], dim='time').sortby('time')


    return AMSR2_stack

