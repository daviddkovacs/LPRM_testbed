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
# from src.config.radiative_14 import sm_i,vod_i,t_i,iterations, number_simulations

number_simulations = 50
sm_i = np.linspace(0.01,1,number_simulations)
vod_i = np.linspace(0.02,1.5,number_simulations)
t_i = np.linspace(270,330,number_simulations)
iterations = np.arange(0, number_simulations,1)

sim_variable = "t"

const_sm = "05"
const_vod = "001"
sim_ds = xr.open_dataset(f"/home/ddkovacs/Desktop/lprm_simulations/{sim_variable}/{sim_variable}_vod001.nc",chunks="auto")
bbox =  [
    -136.41733630337148,
    -51.65735759579918,
    92.18574536660515,
    69.8325684457372
  ]

sim_ds = sim_ds.sel(lat = slice(bbox[3],bbox[1]),
                      lon = slice(bbox[0], bbox[2]))


var_dict = {"sm" :sm_i,
            "vod": vod_i,
            "t": t_i}

m_list = []
c_list = []
for i in range(0, len(iterations)):
# for i in range(20,21):
    try:
        sim_data = sim_ds.isel(i = i)
        TbV_sim = sim_data["TbV_sim"]
        TbH_sim = sim_data["TbH_sim"]
        # sm = sim_data["sm"]
        # opt = sim_data["opt"]
        opt_sim = sim_data["opt_sim"]
        # T = sim_data["T"]

        mpdi = ((TbV_sim-TbH_sim)/(TbV_sim+TbH_sim)).values.flatten()
        Tb_ratio = (TbH_sim/TbV_sim).values.flatten()

        X = Tb_ratio.reshape(-1, 1)
        y = mpdi

        mask = np.isfinite(X[:, 0]) & np.isfinite(y)  # removes NaN + inf
        Xf = X[mask]
        yf = y[mask]

        ransac = HuberRegressor()
        ransac.fit(Xf, yf)
        m = ransac.coef_[0]
        c = ransac.intercept_

        m_list.append(m)
        c_list.append(c)
        print(f"succes data: {i}")

        # plt.figure()
        # plt.hexbin(
        #     Tb_ratio,
        #     mpdi,
        #     gridsize=100,
        #     bins="log",
        #     cmap="magma",
        # )
        # plt.title(f"m:{np.round(m,2)} c:{np.round(c,2)}")
        # # plt.ylim([0,0.07])
        # # plt.xlim([0,1])
        # plt.show()
    except:
        print(f"faulty data: {i}")
        m_list.append(np.nan)
        c_list.append(np.nan)

plt.figure()
plt.plot(var_dict[sim_variable],m_list, label = "gradient")
plt.plot(var_dict[sim_variable],c_list, label ="intercept")
plt.xlabel(sim_variable)
plt.ylabel("m and c")
plt.title(f"const sm {const_sm} vod{const_vod}")
plt.show()

