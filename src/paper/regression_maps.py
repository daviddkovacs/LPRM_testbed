import os
import xarray as xr

from plotter_utils import double_world_plot, mask_rainforest


band1 = "c1"
band2 = "x"
var_lut = {"intercept": {"cbar_range" : [0,100], "title":"",
                         "cmap": "viridis"},
           "slope" :    {"cbar_range" : [0.7,1.1],  "title":"",
                         "cmap": "RdYlGn"} }

var = "intercept"

base_path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/aux_files"

file1 = os.path.join(base_path, f"Daytime_T_aux_{band1.lower()}_MPDI0.01.nc")
file2 = os.path.join(base_path, f"Daytime_T_aux_{band2.lower()}_MPDI0.01.nc")

data1 = xr.open_dataset(file1)[var]
data2 = xr.open_dataset(file2)[var]

filtered_data1 = mask_rainforest(data1)
filtered_data2 = mask_rainforest(data2)

double_world_plot(filtered_data1,
                  filtered_data2,
                  f"C-band",
                  f"X-band",
                  "",
                  var_lut[var]["cmap"], var_lut[var]["cbar_range"],
                      var,
                  )

