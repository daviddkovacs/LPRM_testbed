import matplotlib
import xarray as xr
from LST.load_amsr2 import open_amsr2
from LST.load_modis import open_modis,ndvi_calc
import os
from typing import Literal, List
from config.paths import S3_SLSTR_path, path_bt, MODIS_path, MODIS_geo_path, MODIS_path_local, MODIS_geo_path_local
matplotlib.use("TkAgg")

def OPTICAL_datacube(region: Literal["sahel", "siberia", "midwest", "ceu"],
                         bbox: List[float],
                         time_start = "2024-01-01",
                         time_stop = "2025-01-01",
                         ):
    """
    Main function to obtain MODIS observations, cut to the ROI.
    :param date: Date
    :param bbox: Bound box (lonmin, latmin, lonmax, latmax)
    :param optical_path: Path where SLSTR data is stored. Accepts "SL_2_LST*.SEN3" unpacked folders.
    :param region: Region of MODIS. Currently downloaded:  US Midwest. needs to be checke on GEO network!!
    :return: dictionary with MODIS and AMSR2 datacubes.
    """

    MODIS_path_region = os.path.join(MODIS_path_local, region)

    MODIS_reflectance = open_modis(MODIS_path_region,
                               bbox=bbox,
                               type_of_product="reflectance",
                               time_start=time_start,
                               time_stop=time_stop,
                                   geo_path = MODIS_geo_path)

    MODIS_NDVI = ndvi_calc(MODIS_reflectance["1km Surface Reflectance Band 1"],
                           MODIS_reflectance["1km Surface Reflectance Band 5"])["NDVI"]

    MODIS_LST = open_modis(MODIS_path_region,
                               bbox=bbox,
                               type_of_product="lst",
                               time_start=time_start,
                               time_stop=time_stop)["LST"]


    return MODIS_NDVI, MODIS_LST



def MICROWAVE_datacube(
        bbox: List[float],
        path=path_bt,
        sensor="AMSR2",
        overpass:Literal["day","night","daynight"] = "day",
        time_start="2024-01-01",
        time_stop="2025-01-01",
):
    """
    Function to load AMSR2 Regridded data on the 0.25 or 0.1 CCI grid.
    The swaths are categorized as day or night, thus we handle these differently.
    :param bbox: Bound box (lonmin, latmin, lonmax, latmax)
    :param path: Path where data is found. Edit paths.py for you specific setup.
    :param sensor: Which microwave sensor to use. works best with AMSR2 now.
    :param overpass: Day or Night. See sensor information on the specific Equator overpass time.
    :param time_start: start of time
    :param time_stop: end of time
    :return: Brightness temperatures loaded as an Xarray Datacube.
    """
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

