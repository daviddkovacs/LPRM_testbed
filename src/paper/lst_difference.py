import os
import matplotlib.pyplot as plt
import xarray as xr
from plotter_utils import double_world_plot
from glob import glob

path_amsr2_bt = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/01_resampled_bt/coarse_resolution/AMSR2/day/2024*/*.nc"
reg_parameters_path  =  "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/02_aux/daytime_optimisation/lst_regression_parameters/coarse_resolution/daytime_lst_regression_25km.nc"

files_amsr2 = glob(path_amsr2_bt)
bt_stack = xr.open_mfdataset(files_amsr2,decode_timedelta=  False,)["bt_36.5V"]
reg_param = xr.open_datatree(reg_parameters_path)
##
holmes = (bt_stack * 0.893 + 44.8).compute()
holmes_mean = holmes.mean(dim="time")
##

slope_c1 = reg_param["C1"]["slope"]
intercept_c1 = reg_param["C1"]["intercept"]
daytime_c1 = (bt_stack * slope_c1 + intercept_c1).compute().mean(dim="time")
diff_c1 = daytime_c1 - holmes_mean

slope_x = reg_param["X"]["slope"]
intercept_x = reg_param["X"]["intercept"]
daytime_x = (bt_stack * slope_x + intercept_x).compute().mean(dim="time")
diff_x = daytime_x - holmes_mean


##


double_world_plot(diff_c1,diff_x, r"C1-band", r"X-band",
                  r"2024 Mean T$_{\mathrm{eff}}$ Difference (Regression − Holmes)",
                  "coolwarm",
                  [-10,10],
                  r"$\Delta$ Temperature [K]"
                  )

