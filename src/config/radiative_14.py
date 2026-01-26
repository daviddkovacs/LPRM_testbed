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

sat_band = "C1"
frequencies={'C1': 6.9, 'C2': 7.3, 'X': 10.7,'KU': 18.7, 'K': 23.8, 'KA': 36.5}
sat_sensor = "amsr2"
specs = get_specs(sat_sensor.upper())
params = get_lprm_parameters_for_frequency(sat_band, specs.incidence_angle)

def auxdata(string_type):
    file =f"/home/ddkovacs/Desktop/soil_maps/auxiliary_data_{string_type}_25km"

    tif_data = gdal.Open(file, gdal.GA_ReadOnly)
    array = tif_data.GetRasterBand(1).ReadAsArray().astype(float)
    return array

year = "2024"
bt_path = os.path.join(path_bt,"day",f"{year}*", f"*day_{year}*.nc")
bt_files = glob.glob(bt_path)

bt_data = xr.open_dataset(bt_files[110], decode_timedelta=False)
da_d = bt_data["bt_23.8H"].dims
da_c = bt_data["bt_23.8H"].drop_vars("time").coords

lats = 720
lons = 1440

number_simulations = 50
sm_i = np.linspace(0.01,1,number_simulations)
vod_i = np.linspace(0.02,1.5,number_simulations)
t_i = np.linspace(270,330,number_simulations)
iterations = np.arange(0, number_simulations,1)

sm_stack =   sm_i[None, None,:] * np.ones((lats,lons,len(sm_i)))
vod_stack =   vod_i[None, None,:] * np.ones((lats,lons,len(vod_i)))
t_stack =   t_i[None, None,:] * np.ones((lats,lons,len(vod_i)))

sm_cons = 0.01
vod_cons = 0.01
t_cons = 313

i_variable = "sm"

da_list = []
da_dict = {}
opt_sim_list = []
for i in range(0,len(iterations)):
    print(i)
    if i_variable.lower() == "sm":
        sm_slice = sm_stack[:,:,i]
        vod_slice = np.full((lats,lons), vod_cons)
        t_slice = np.full((lats,lons), t_cons)

    elif i_variable.lower() == "vod":
        sm_slice =  np.full((lats,lons), sm_cons)
        vod_slice = vod_stack[:,:,i]
        t_slice = np.full((lats,lons), t_cons)

    elif i_variable.lower() == "t":
        sm_slice =  np.full((lats,lons), sm_cons)
        vod_slice = np.full((lats, lons), vod_cons)
        t_slice = t_stack[:,:,i]

    else:
        raise Exception

    TbH_sim, TbV_sim, opt_sim = radiative_transfer(
        sm_slice,
        vod_slice,
        t_slice,
        auxdata( "SND"),  # fixed
        auxdata( "CLY"),  # fixed
        auxdata( "BLD"),  # fixed
        params.Q,  # fixed
        params.w,  # fixed
        0,  # fixed
        specs.incidence_angle[0],  # fixed
        params.h1,  # fixed
        params.h2,  # fixed
        params.vod_Av,  # fixed
        params.vod_Bv,  # fixed
        float(get_specs(sat_sensor.upper()).frequencies[sat_band.upper()]),  # fixed
        params.temp_freeze,  # fixed
        False,
        None,
        # T_theor=Teff.values.astype('double'),
        # Theory_select = selector
    )
    da = xr.Dataset(
        data_vars=dict(
            TbH_sim=(("lat", "lon"), TbH_sim),
            TbV_sim=(("lat", "lon"), TbV_sim),
            # sm=(("lat", "lon"), sm_slice ),
            # opt=(("lat", "lon"), vod_slice),
            opt_sim=(("lat", "lon"), opt_sim),
            # T_soil=(("lat", "lon"), T_soil),
            # T_canopy=(("lat", "lon"), T_canopy),
        ),
        coords=da_c,
    ).expand_dims(i=[i])
    da_list.append(da)
    opt_sim_list.append(np.nanmean(opt_sim))
sim_ds = xr.concat(da_list,dim = "i")

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
        # T_soil = sim_data["T_soil"]
        # T_canopy = sim_data["T_canopy"]
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
        m = np.where(abs(m)>0.1,m,np.nan)
        c = np.where(abs(c)>0.1,c,np.nan)
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
    except Exception as e:

        print(f"faulty data: {e}")
        m_list.append(np.nan)
        c_list.append(np.nan)

title  = {"vod": f"constants sm: {sm_cons} T: {t_cons}",
          "sm" : f"constants vod: {vod_cons} T: {t_cons}",
          "t" : f"constants sm: {sm_cons} vod: {vod_cons}",}
plt.figure()
plt.plot(var_dict[i_variable],m_list, label = "gradient")
plt.plot(var_dict[i_variable],opt_sim_list, label = "opt")
plt.plot(var_dict[i_variable],c_list, label ="intercept")
plt.xlabel(i_variable)
plt.legend()
plt.ylabel("m and c")
plt.title(title[i_variable])
plt.ylim([-0.85,0.85])
plt.show()




#
#
# comp = dict(zlib=True, complevel=5, shuffle=True, dtype='float32')
# encoding = {var : comp for var in dataset.var() }
#
# dataset.to_netcdf(f"/home/ddkovacs/Desktop/lprm_simulations/{i_variable}/{i_variable}_vod{vod_cons}.nc",encoding=encoding)
