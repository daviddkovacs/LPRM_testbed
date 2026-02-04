import os.path
from typing import Literal
from config.paths import SLSTR_path, path_bt
from LST.SLSTR_utils import (preprocess_slstr,
                         open_sltsr,
                         open_amsr2,
                         crop2roi,
                         threshold_ndvi,
                         get_edges,
                         compare_temperatures
                             )
from datetime import datetime, timedelta
from plot_functions import (plot_lst,
                            temps_plot,
                            LST_plot_params, NDVI_plot_params,AMSR2_plot_params, plot_amsr2)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def preprocess_datacubes(SLSTR, AMSR2, date, bbox):

    # preprocess SLSTR
    SLSTR_obs = SLSTR.sel(time=date, method="nearest")

    # We select SLSTR's observation to get AMSR2. the frequency of obs for AMSR2 is higher.
    AMSR2_obs = AMSR2.sortby('time').sel(time=SLSTR_obs.time.dt.floor("d"), method="nearest")
    AMSR2_roi = crop2roi(AMSR2_obs.compute(), bbox)
    AMSR2_roi["TSURF"] = AMSR2_roi["bt_36.5V"] * 0.893 + 44.8

    AMSR2_bbox = [get_edges(AMSR2_roi.lon.values).min(),
                  get_edges(AMSR2_roi.lat.values).min(),
                  get_edges(AMSR2_roi.lon.values).max(),
                  get_edges(AMSR2_roi.lat.values).max()]

    SLSTR_roi = crop2roi(SLSTR_obs.compute(), AMSR2_bbox)

    plot_lst(left_da=SLSTR_obs["LST"],
             right_da=SLSTR_obs["NDVI"],
             left_params=LST_plot_params,
             right_params=NDVI_plot_params,
             bbox=bbox)

    # plot_lst(left_da=SLSTR_roi["LST"],
    #          right_da=SLSTR_roi["NDVI"],
    #          left_params=LST_plot_params,
    #          right_params=NDVI_plot_params)

    return {"SLSTR": SLSTR_roi, "AMSR2": AMSR2_roi}


def SLSTR_AMSR2_datacubes( region : Literal["sahel", "siberia", "midwest"],
                           SLSTR_path = SLSTR_path,
                           AMSR2_path = path_bt,):
    """
    Main function to obtain SLSTR and AMSR2 observations, cut to the ROI.
    :param date: Date
    :param bbox: Bound box (lonmin, latmin, lonmax, latmax)
    :param SLSTR_path: Path where SLSTR data is stored. Accepts "SL_2_LST*.SEN3" unpacked folders.
    :param AMSR2_path: Path where AMSR2 brightness temperatures are stored
    :param region: Region of SLSTR. Currently downloaded: Sahel, Siberia and US Midwest
    :return: dictionary with SLSTR and AMSR2 datacubes.
    """
    SLSTR_path_region = os.path.join(SLSTR_path,region)

    NDVI = open_sltsr(path=SLSTR_path_region,
                   subdir_pattern=f"S3A_SL_2_LST____*",
                   date_pattern=r'___(\d{8})T(\d{4})',
                   variable_file="LST_ancillary_ds.nc",
                        )
    LST= open_sltsr(path=SLSTR_path_region,
                   subdir_pattern=f"S3A_SL_2_LST____*",
                   date_pattern=r'___(\d{8})T(\d{4})',
                   variable_file="LST_in.nc",
                        )
    SLSTR = preprocess_slstr(NDVI, LST, SLSTR_path_region)

    AMSR2 = open_amsr2(path=AMSR2_path,
                       sensor="AMSR2",
                       overpass="day",
                       subdir_pattern=f"20*",
                       file_pattern="amsr2_l1bt_*.nc",
                       date_pattern=r"_(\d{8})_",
                       time_start="2024-01-01",
                       time_stop="2025-01-01",
                       )


    return  {"SLSTR" : SLSTR, "AMSR2" : AMSR2,}




if __name__=="__main__":
    DATACUBES = SLSTR_AMSR2_datacubes(region="sahel")
##
    date = "2024-01-10"

    bbox = [
    -3.114992771726577,
    12.58077857296621,
    -0.12847188822379962,
    14.320359668899599
  ]

    ndvi_threshold  = 0.5

    DATACUBES_L2 = preprocess_datacubes(SLSTR=DATACUBES["SLSTR"],
                                        AMSR2 = DATACUBES["AMSR2"],
                                        date=date,
                                        bbox=bbox)

    SLSTR_LST = DATACUBES_L2["SLSTR"]["LST"]
    SLSTR_NDVI = DATACUBES_L2["SLSTR"]["NDVI"]
    AMSR2_LST = DATACUBES_L2["AMSR2"]["TSURF"]

    soil_temp, veg_temp = threshold_ndvi(lst = SLSTR_LST,
                                         ndvi = SLSTR_NDVI,
                                         ndvi_thres = ndvi_threshold)

    plot_amsr2(AMSR2_LST,AMSR2_plot_params)

    df = compare_temperatures(soil_temp,veg_temp,AMSR2_LST)
    _df = df.sort_values(by="tsurf_ka").dropna(subset="tsurf_ka")
    temps_plot(_df)

