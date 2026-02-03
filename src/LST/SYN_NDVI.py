import pandas as pd

from LST.NDVI_utils import snow_filtering, get_edges
from config.paths import SLSTR_path, path_bt
import xarray as xr
import numpy as np
from NDVI_utils import (open_sltsr,
                        open_amsr2,
                        filter_empty_var,
                        crop2roi,
                        threshold_ndvi,
                        cloud_filtering,
                        slstr_pixels_in_amsr2,
                        subset_statistics,
                        get_edges,
                        binning_smaller_pixels)
from plot_functions import (plot_lst,
                            plot_amsr2,
                            boxplot_soil_veg, temps_plot, LST_plot_params, NDVI_plot_params, AMSR2_plot_params)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


if __name__=="__main__":

    NDVI = open_sltsr(path=SLSTR_path,
                   subdir_pattern=f"S3A_SL_2_LST____*",
                   date_pattern=r'___(\d{8})T(\d{4})',
                   variable_file="LST_ancillary_ds.nc",
                   # bbox= bbox
                        )

    LST= open_sltsr(path=SLSTR_path,
                   subdir_pattern=f"S3A_SL_2_LST____*",
                   date_pattern=r'___(\d{8})T(\d{4})',
                   variable_file="LST_in.nc",
                   # bbox= bbox
                        )

    AMSR2 = open_amsr2(path=path_bt,
                       sensor="AMSR2",
                       overpass="day",
                       subdir_pattern=f"20*",
                       file_pattern="amsr2_l1bt_*.nc",
                       date_pattern=r"_(\d{8})_",
                       time_start="2024-05-01",
                       time_stop="2025-07-01",
                       )

    _SLSTR = xr.merge([NDVI,LST])[["LST","NDVI"]]

    _SLSTR = cloud_filtering(_SLSTR) # Mask clouds (strict)
    _SLSTR = snow_filtering(_SLSTR) # Mask clouds (strict)

    SLSTR = filter_empty_var(_SLSTR, "NDVI") # Filter empty NDVI obs

    ##
    date = "2024-08-25"

    bbox = [
    -106.50714078246102,
    35.262436997113625,
    -100.81621233002745,
    38.02600079214329
  ]
    ndvi_thres =0.5

    AMSR2_obs = AMSR2.sortby('time').sel(time=date, method="nearest")
    AMSR2_obs = crop2roi(AMSR2_obs.compute(),bbox)
    TSURF = AMSR2_obs["bt_36.5V"] * 0.893 + 44.8

    AMSR2_bbox = [get_edges(TSURF.lon.values).min(),
                  get_edges(TSURF.lat.values).min(),
                  get_edges(TSURF.lon.values).max(),
                  get_edges(TSURF.lat.values).max()]

    SLSTR_obs = SLSTR.sel(time=date, method="nearest")
    SLSTR_obs = crop2roi(SLSTR_obs.compute(),AMSR2_bbox)

    soil_temp, veg_temp = threshold_ndvi(lst = SLSTR_obs["LST"], ndvi = SLSTR_obs["NDVI"] ,ndvi_thres = ndvi_thres)

    plot_lst(left_da = SLSTR_obs["LST"],
             right_da = SLSTR_obs["NDVI"],
             left_params=LST_plot_params,
             right_params= NDVI_plot_params)

    # plot_amsr2(TSURF,AMSR2_plot_params)

##
    # szerintem ebbol csinaljak majd egy functiont hetfon

    veg_mean_list = []
    veg_std_list = []

    soil_mean_list = []
    soil_std_list = []
    TSURF_list = []

    bin_dict = binning_smaller_pixels(SLSTR_obs["NDVI"], TSURF)

    for targetlat in range(0, bin_dict["lats"].max()):
        for targetlon in range(0,bin_dict["lons"].max()):

            soil_subset = slstr_pixels_in_amsr2(soil_temp,
                                  bin_dict,
                                  targetlat,
                                  targetlon)

            veg_subset =  slstr_pixels_in_amsr2(veg_temp,
                                  bin_dict,
                                  targetlat,
                                  targetlon)

            soil_mean_list.append(subset_statistics(soil_subset)[1]["mean"])
            soil_std_list.append(subset_statistics(soil_subset)[1]["std"])

            veg_mean_list.append(subset_statistics(veg_subset)[1]["mean"])
            veg_std_list.append(subset_statistics(veg_subset)[1]["std"])

            TSURF_subset = TSURF.isel(lat=targetlat,lon=targetlon)
            TSURF_list.append(TSURF_subset.values)

            # plt.figure()
            # TSURF.plot()
            # plt.show()
            # plt.figure()
            # soil_subset.plot(x = "lon",y = "lat")
            # plt.show()
            # plt.figure()
            # veg_subset.plot(x = "lon",y = "lat")
            # plt.show()

    df_temps = pd.DataFrame({"veg_mean": veg_mean_list,
                             "veg_std": veg_std_list,
                             "soil_mean" :soil_mean_list,
                             "soil_std" :soil_std_list,
                             "tsurf_ka": TSURF_list,
                             }).sort_values(by="tsurf_ka")
    temps_plot(df_temps)

##
    # soil_plot_params = {"x": "lon",
    #                    "y":"lat",
    #                    "cmap":"coolwarm",
    #                    "cbar_kwargs":{'label': 'Soil LST [K]'},
    #                    "vmin":290,
    #                     "vmax": 320,
    #                     "title": "Soil (NDVI<0.3) LST"
    #                    }
    # veg_plot_params = {
    #                     "x":"lon",
    #                     "y":"lat",
    #                     "cmap":"coolwarm",
    #                     "cbar_kwargs":{'label':"NDVI [-]"},
    #                     "vmin" : 290,
    #                     "vmax" : 320,
    #                     "title" :"Veg. (NDVI>0.3) LST"
    #                    }
    #
    # plot_lst(left_da = veg_temp,
    #          right_da = soil_temp,
    #          left_params=veg_plot_params,
    #          right_params= soil_plot_params)

##
