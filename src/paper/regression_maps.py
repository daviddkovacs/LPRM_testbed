import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from glob import glob
from plotter_utils import double_world_plot

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

data1_filtered = data1.where(data1>10)
data2_filtered = data2.where(data1>10)

double_world_plot(data1_filtered,
                  data2_filtered,
                  f"{band1.upper()}-band",
                  f"{band2.upper()}-band",
                  "",
                  var_lut[var]["cmap"], var_lut[var]["cbar_range"],
                      var,
                  )

