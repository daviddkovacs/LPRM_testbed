import matplotlib.pyplot as plt
from scipy.stats import alpha
from config.paths import SLSTR_path
import xarray as xr
import pandas as pd
import numpy as np
from NDVI_utils import open_sltsr, filter_empty, plot_lst, crop2roi

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

    _SLSTR = xr.merge([NDVI,LST])[["LST","NDVI"]]
    SLSTR = filter_empty(_SLSTR,"NDVI")

    ##
    date = "2024-04-30"

    bbox = [
    -104.72584198593596,
    36.51867771410325,
    -104.0223717289054,
    37.34538593805205
  ]

    SLSTR_obs = SLSTR.sel(time=date, method="nearest")
    SLSTR_obs = crop2roi(SLSTR_obs.compute(),bbox)

    veg_temp = xr.where(SLSTR_obs["NDVI"]>0.3,SLSTR_obs["LST"], np.nan)
    soil_temp = xr.where(SLSTR_obs["NDVI"]<0.3,SLSTR_obs["LST"], np.nan)

    LST_plot_params = {"x": "lon",
                       "y":"lat",
                       "cmap":"coolwarm",
                       "cbar_kwargs":{'label': 'LST [K]'},
                       "vmin":273,
                       "title": "LST"
                       }
    NDVI_plot_params = {
                        "x":"lon",
                        "y":"lat",
                        "cmap":"YlGn",
                        "cbar_kwargs":{'label':"NDVI [-]"},
                        "vmin" : 0,
                        "vmax" : 0.6,
                        "title" :"NDVI"
                       }

    plot_lst(left_da = SLSTR_obs["LST"],
             right_da = SLSTR_obs["NDVI"],
             left_params=LST_plot_params,
             right_params= NDVI_plot_params)

##

    soil_plot_params = {"x": "lon",
                       "y":"lat",
                       "cmap":"coolwarm",
                       "cbar_kwargs":{'label': 'Soil LST [K]'},
                       "vmin":273,
                        "vmax": 320,
                        "title": "Soil (NDVI<0.3) LST"
                       }
    veg_plot_params = {
                        "x":"lon",
                        "y":"lat",
                        "cmap":"coolwarm",
                        "cbar_kwargs":{'label':"NDVI [-]"},
                        "vmin" : 273,
                        "vmax" : 320,
                        "title" :"Veg. (NDVI>0.3) LST"
                       }

    plot_lst(left_da = veg_temp,
             right_da = soil_temp,
             left_params=veg_plot_params,
             right_params= soil_plot_params)

##

hist_soil = soil_temp.values.flatten()[~np.isnan(soil_temp.values.flatten())]
hist_veg = veg_temp.values.flatten()[~np.isnan(veg_temp.values.flatten())]

plt.hist(hist_soil,
         bins =200,
         alpha = 0.8,
         label = "T_soil (NDVI<0.3)",
         color = "brown"
         )
plt.hist(hist_veg,
         bins =200,
         alpha = 0.9,
         label="T_vegetation (NDVI<0.3)",
         color = "green"
         )
plt.xlabel("T [K]")
plt.ylabel("freq")
plt.title("Soil/Veg. Temp")
plt.legend(loc ="upper left")
plt.show()