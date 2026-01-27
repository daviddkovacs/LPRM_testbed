# import os
# import glob
# from config.paths import path_bt
# import matplotlib
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# from lprm.retrieval.lprm_v6_1.parameters import get_lprm_parameters_for_frequency
# from lprm.satellite_specs import get_specs
# from simulator.radiative_transfer_lprm import radiative_transfer
# matplotlib.use("TkAgg")
# from osgeo import gdal
#
#
# sat_band = "C1"
# frequencies={'C1': 6.9, 'C2': 7.3, 'X': 10.7,'KU': 18.7, 'K': 23.8, 'KA': 36.5}
# sat_sensor = "amsr2"
# specs = get_specs(sat_sensor.upper())
# params = get_lprm_parameters_for_frequency(sat_band, specs.incidence_angle)
#
# def auxdata(string_type):
#     file =f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/02_aux_data/coarse_resolution/lprm_v6/soil_content/auxiliary_data_{string_type}_25km"
#
#     tif_data = gdal.Open(file, gdal.GA_ReadOnly)
#     array = tif_data.GetRasterBand(1).ReadAsArray().astype(float)
#     return array
#
# ##
# year = "2024"
# bt_path = os.path.join(path_bt,"day",f"{year}*", f"*day_{year}*.nc")
# bt_files = glob.glob(bt_path)
#
# bt_data = xr.open_dataset(bt_files[110], decode_timedelta=False)
# da_d = bt_data["bt_23.8H"].dims
# da_c = bt_data["bt_23.8H"].drop("time").coords
#
# lats = 720
# lons = 1440
#
# sm_i = np.arange(0.01,1,0.01)
# vod_i = np.arange(0.02,2,0.02)
# t_i = np.linspace(270,330,99)
# iterations = len(sm_i)
#
# sm_stack =   sm_i[None, None,:] * np.ones((lats,lons,len(sm_i)))
# vod_stack =   vod_i[None, None,:] * np.ones((lats,lons,len(vod_i)))
# t_stack =   t_i[None, None,:] * np.ones((lats,lons,len(vod_i)))
#
# sm_cons = 0.5
# vod_cons = 0.7
# t_cons = 293
#
# i_variable = "sm"
#
# for i in range(0,2):
#     print(i)
#     if i_variable.lower() == "sm":
#         sm_slice = sm_stack[:,:,i]
#         vod_slice = np.full((lats,lons), vod_cons)
#         t_slice = np.full((lats,lons), t_cons)
#
#     elif i_variable.lower() == "vod":
#         sm_slice =  np.full((lats,lons), sm_cons)
#         vod_slice = vod_stack[:,:,i]
#         t_slice = np.full((lats,lons), t_cons)
#
#     elif i_variable.lower() == "t":
#         sm_slice =  np.full((lats,lons), sm_cons)
#         vod_slice = np.full((lats, lons), vod_cons)
#         t_slice = t_stack[:,:,i]
#
#     else:
#         raise Exception
#
#     TbH_sim, TbV_sim,smrun2, opt, T = radiative_transfer(
#         sm_slice,
#         vod_slice,
#         t_slice,
#         auxdata( "SND"),  # fixed
#         auxdata( "CLY"),  # fixed
#         auxdata( "BLD"),  # fixed
#         params.Q,  # fixed
#         params.w,  # fixed
#         0,  # fixed
#         specs.incidence_angle[0],  # fixed
#         params.h1,  # fixed
#         params.h2,  # fixed
#         params.vod_Av,  # fixed
#         params.vod_Bv,  # fixed
#         float(get_specs(sat_sensor.upper()).frequencies[sat_band.upper()]),  # fixed
#         params.temp_freeze,  # fixed
#         False,
#         None,
#         # T_theor=THolmes.values.astype('double'),
#         # Theory_select = selector
#     )
#     tb_h = xr.DataArray(
#         data=TbH_sim,
#         dims=("lon","lat"),
#         coords=da_c,
#         name=f'TbH_sim'
#     )
#     x =1
#
#
# plt.figure()
# tb_h.plot()
# plt.show()