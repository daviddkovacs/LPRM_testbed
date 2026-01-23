import os
import glob
from config.paths import path_bt, path_lprm
import pandas as pd
from readers.Sat import BTData, LPRMData
import matplotlib
import numpy as np
from sklearn.linear_model import HuberRegressor

import xarray as xr
import matplotlib.pyplot as plt
from lprm.retrieval.lprm_v6_1.parameters import get_lprm_parameters_for_frequency
from lprm.satellite_specs import get_specs
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from shapely.geometry import LineString,  Point
from lprm.retrieval.lprm_v6_1.run_lprmv6 import load_band_from_ds
from lprm.retrieval.lprm_general import load_aux_file
from utilities.radiative_transfer_lprm import radiative_transfer
matplotlib.use("TkAgg")
from osgeo import gdal


sim_variable = "sm"
sim_data = xr.open_dataset(f"/home/ddkovacs/Desktop/lprm_simulations/{sim_variable}.nc",chunks="auto")
bbox =  [
    -136.41733630337148,
    -51.65735759579918,
    92.18574536660515,
    69.8325684457372
  ]
sim_data = sim_data.sel(lat = slice(bbox[3],bbox[1]),
                      lon = slice(bbox[0], bbox[2]))

iteration = 50
m_list = []
c_list = []

sim_data = sim_data.isel(i = iteration)
TbV_sim = sim_data["TbV_sim"]
TbH_sim = sim_data["TbH_sim"]
sm = sim_data["sm"]
opt = sim_data["opt"]
opt_sim = sim_data["opt_sim"]
T = sim_data["T"]

mpdi = ((TbV_sim-TbH_sim)/(TbV_sim+TbH_sim)).values.flatten()
Tb_ratio = (TbH_sim/TbV_sim).values.flatten()

ransac = HuberRegressor()
ransac.fit(Tb_ratio.reshape(-1, 1), mpdi)
line_x_ransac = np.arange(Tb_ratio.min(), Tb_ratio.max(), 0.01)[:, np.newaxis]
line_y_ransac = ransac.predict(line_x_ransac)
m, c = np.polyfit(line_x_ransac.ravel(), line_y_ransac, 1)



