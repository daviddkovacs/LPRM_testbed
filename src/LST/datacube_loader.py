import matplotlib
import xarray as xr
from LST.load_amsr2 import open_amsr2
from LST.load_slstr import open_sltsr
from LST.load_modis import open_modis,ndvi_calc
matplotlib.use("TkAgg")

import os
from typing import Literal, List
from config.paths import S3_SLSTR_path, path_bt, MODIS_path_local, MODIS_geo_path_local




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

