import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from glob import glob
from plotter_utils import double_world_plot

band1 = "x"
band2 = "c1"
var_lut = {"intercept": {"cbar_range" : [0,100], "title":""},
           "slope" :    {"cbar_range" : [0,1.2],    "title":""} }

var = "slope"


base_path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/aux_files"

file1 = os.path.join(base_path, f"Daytime_T_aux_{band1.lower()}_MPDI0.01.nc")
file2 = os.path.join(base_path, f"Daytime_T_aux_{band2.lower()}_MPDI0.01.nc")


data1 = xr.open_dataset(file1)[var]
data2 = xr.open_dataset(file2)[var]



double_world_plot(data2,
                  data1,
                  f"{band1.upper()}-band",
                  f"{band2.upper()}-band",
                  "",
                  "viridis", var_lut[var]["cbar_range"],
                      var,
                  )

