import matplotlib
import matplotlib.pyplot as plt
import bottleneck
from LST.datacube_utilities import crop2roi, get_edges
from LST.load_amsr2 import open_amsr2
from LST.load_slstr import open_sltsr
from LST.load_modis import open_modis,ndvi_calc
from LST.plot_functions import plot_modis_comparison
matplotlib.use("TkAgg")

import os
from typing import Literal, List
from config.paths import S3_SLSTR_path, path_bt, MODIS_path


# ---------------------------------------
# DATACUBE PROCESSORS
def temporal_subset_dc(SLSTR, AMSR2, date):
    """
    Select the closest date to SLSTR, and thus select this date to access AMSR2
    """
    # SLSTR["time"] = SLSTR.time.sortby("time")
    SLSTR = SLSTR.drop_duplicates(dim="time")
    SLSTR_obs = SLSTR.sel(time=date, method="nearest")

    # We select SLSTR's observation to get AMSR2. the frequency of obs for AMSR2 is higher.
    AMSR2_obs = AMSR2.sortby('time').sel(time=SLSTR_obs.time.dt.floor("d"), method="nearest")

    return {"SLSTR": SLSTR_obs, "AMSR2": AMSR2_obs}


def spatial_subset_dc(SLSTR, AMSR2,  bbox):
    """
    SLSTR is cut to the full spatial extent of AMSR2.
    Both AMSR2 and SLSTR cropped to bbox
    """
    AMSR2 = crop2roi(AMSR2, bbox)
    res = AMSR2.attrs["resolution"]

    AMSR2_bbox = [get_edges(AMSR2.lon.values, res).min(),
                  get_edges(AMSR2.lat.values, res).min(),
                  get_edges(AMSR2.lon.values, res).max(),
                  get_edges(AMSR2.lat.values, res).max()]

    SLSTR_roi = crop2roi(SLSTR, AMSR2_bbox)

    return {"SLSTR": SLSTR_roi, "AMSR2": AMSR2}


def OPTI_AMSR2_DATACUBES(region: Literal["sahel", "siberia", "midwest", "ceu"],
                         bbox: List[float],
                         sensor: Literal["MODIS", "SLSTR"],
                         AMSR2_path = path_bt,
                         time_start = "2024-01-01",
                         time_stop = "2025-01-01",
                         ):
    """
    Main function to obtain SLSTR and AMSR2 observations, cut to the ROI.
    :param date: Date
    :param bbox: Bound box (lonmin, latmin, lonmax, latmax)
    :param optical_path: Path where SLSTR data is stored. Accepts "SL_2_LST*.SEN3" unpacked folders.
    :param AMSR2_path: Path where AMSR2 brightness temperatures are stored
    :param region: Region of SLSTR. Currently downloaded: Sahel, Siberia and US Midwest
    :return: dictionary with SLSTR and AMSR2 datacubes.
    """

    if sensor.upper() == "SLSTR":

        SLSTR_path_region = os.path.join(S3_SLSTR_path, region)
        optcial_stack = open_sltsr(SLSTR_path_region,
                                   time_start = time_start,
                                   time_stop = time_stop,
                                   bbox=bbox
                                   )

    elif sensor.upper() == "MODIS":
        MODIS_path_region = os.path.join(MODIS_path, region)

        MODIS_reflectance = open_modis(MODIS_path_region,
                                   bbox=bbox,
                                   type_of_product="reflectance",
                                   time_start=time_start,
                                   time_stop=time_stop)

        MODIS_NDVI = ndvi_calc(MODIS_reflectance["1km Surface Reflectance Band 1"],
                               MODIS_reflectance["1km Surface Reflectance Band 5"])

        MODIS_LST = open_modis(MODIS_path_region,
                                   bbox=bbox,
                                   type_of_product="lst",
                                   time_start=time_start,
                                   time_stop=time_stop)

        plotdate = "2018-01-14T16:00"
        plot_modis_comparison(MODIS_NDVI, MODIS_LST["LST"], ndvi_time=plotdate,
                              lst_time=plotdate)

    else:
        optcial_stack = None



    AMSR2_cropped_stack = open_amsr2(path=AMSR2_path,
                                     sensor="AMSR2",
                                     overpass="day",
                                     subdir_pattern=f"20*",
                                     file_pattern="amsr2_l1bt_*.nc",
                                     date_pattern=r"_(\d{8})_",
                                     time_start=time_start,
                                     time_stop=time_stop,
                                     resolution = "coarse_resolution",
                                     bbox=bbox
                                     )


    return  {"SLSTR" : optcial_stack.compute(), "AMSR2" : AMSR2_cropped_stack.compute(),}














