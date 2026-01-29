from LST.NDVI_utils import snow_filtering
from config.paths import SLSTR_path, path_bt
import xarray as xr
import numpy as np
from NDVI_utils import (open_sltsr,
                        open_amsr2,
                        filter_empty_var,
                        plot_lst, crop2roi,
                        filternan,
                        cloud_filtering)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

##


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
                       time_start="2024-01-01",
                       time_stop="2025-01-01",
                       )

    _SLSTR = xr.merge([NDVI,LST])[["LST","NDVI"]]

    _SLSTR = cloud_filtering(_SLSTR) # Mask clouds (strict)
    _SLSTR = snow_filtering(_SLSTR) # Mask clouds (strict)

    SLSTR = filter_empty_var(_SLSTR, "NDVI") # Filter empty NDVI obs

    ##
    date = "2024-07-25"

    bbox = [
    -107.79360536401147,
    33.308332423254896,
    -97.7708702338834,
    40.160182398589114
  ]
    ndvi_thres =0.3
    SLSTR_obs = SLSTR.sel(time=date, method="nearest")
    SLSTR_obs = crop2roi(SLSTR_obs.compute(),bbox)

    veg_temp = xr.where(SLSTR_obs["NDVI"]>ndvi_thres,SLSTR_obs["LST"], np.nan)
    soil_temp = xr.where(SLSTR_obs["NDVI"]<ndvi_thres,SLSTR_obs["LST"], np.nan)

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
                       "vmin":290,
                        "vmax": 320,
                        "title": "Soil (NDVI<0.3) LST"
                       }
    veg_plot_params = {
                        "x":"lon",
                        "y":"lat",
                        "cmap":"coolwarm",
                        "cbar_kwargs":{'label':"NDVI [-]"},
                        "vmin" : 290,
                        "vmax" : 320,
                        "title" :"Veg. (NDVI>0.3) LST"
                       }

    plot_lst(left_da = veg_temp,
             right_da = soil_temp,
             left_params=veg_plot_params,
             right_params= soil_plot_params)

##

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(filternan(soil_temp),
         bins=200,
         alpha=0.8,
         label=f"$T_{{soil}}$ (NDVI < {ndvi_thres})",
         color="brown")
ax1.hist(filternan(veg_temp),
         bins=200,
         alpha=0.7,
         label=f"$T_{{vegetation}}$ (NDVI > {ndvi_thres})",
         color="green")
ax1.set_xlabel("$T$ [K]")
ax1.set_ylabel("frequency")
ax1.set_title("Temp Distribution")
ax1.legend(loc="upper left")

data_to_plot = [filternan(soil_temp), filternan(veg_temp)]
bp = ax2.boxplot(data_to_plot,
                 patch_artist=True,
                 showfliers = False,
                 tick_labels=[f"Soil", f"Veg"])

colors = ["brown", "green"]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel("$T$ [K]")
ax2.set_title("Soil/Veg. Temp Boxplot")

plt.tight_layout()
plt.show()
