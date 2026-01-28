import os
import glob
from dask.array import block
from config.paths import path_bt
import matplotlib
import numpy as np
from sklearn.linear_model import HuberRegressor
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lprm.retrieval.lprm_v6_1.parameters import get_lprm_parameters_for_frequency
from lprm.satellite_specs import get_specs
from simulator.radiative_transfer_lprm import radiative_transfer
from osgeo import gdal
from utilities.retrieval_helpers import get_coords
matplotlib.use("TkAgg")

sat_band = "C1"
frequencies={'C1': 6.9, 'C2': 7.3, 'X': 10.7,'KU': 18.7, 'K': 23.8, 'KA': 36.5}
sat_sensor = "amsr2"
specs = get_specs(sat_sensor.upper())
params = get_lprm_parameters_for_frequency(sat_band, specs.incidence_angle)

def auxdata(string_type):
    file =f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/02_aux/soil_maps/coarse_resolution/lprm_v6/soil_content/auxiliary_data_{string_type}_25km"

    tif_data = gdal.Open(file, gdal.GA_ReadOnly)
    array = tif_data.GetRasterBand(1).ReadAsArray().astype(float)
    return array

lats = 720
lons = 1440

bbox =  [
    -136.41733630337148,
    -51.65735759579918,
    92.18574536660515,
    69.8325684457372
  ]

number_simulations = 50

sm_i = np.linspace(0.01,1,number_simulations)
vod_i = np.linspace(0.02,1.5,number_simulations)
t_i = np.linspace(270,330,number_simulations)
iterations = np.arange(0, number_simulations,1)

sm_dummy = xr.DataArray(
    data= sm_i[None, None,:] * np.ones((lats,lons,len(sm_i))), # sm_stack
    dims=["lat","lon","i"],
    coords= dict(
        lat =  get_coords()["coords"]["lat"],
        lon = get_coords()["coords"]["lon"],
        i =  sm_i
    )
)

vod_dummy = xr.DataArray(
    data= vod_i[None, None,:] * np.ones((lats,lons,len(vod_i))), # vod_stack
    dims=["lat","lon","i"],
    coords= dict(
        lat =  get_coords()["coords"]["lat"],
        lon = get_coords()["coords"]["lon"],
        i =  vod_i
    )
)

t_dummy = xr.DataArray(
    data= t_i[None, None,:] * np.ones((lats,lons,len(vod_i))), # t_stack
    dims=["lat","lon","i"],
    coords= dict(
        lat =  get_coords()["coords"]["lat"],
        lon = get_coords()["coords"]["lon"],
        i =  t_i
    )
)

sm_cons = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.99]
vod_cons = [0.01,0.1, 0.2, 0.4, 0.6, 0.8, 1]
t_cons = 293

i_variable = "vod"

opt_sim_list = []

const_tbv_dict = {}
const_tbh_dict = {}
const_m_dict = {}
const_c_dict = {}

for const_variable_value in sm_cons:
    print(f"SM value to be: {str(const_variable_value)}")

    tbv_list = []
    tbh_list = []
    m_list = []
    c_list = []

    for i in range(0,len(iterations)):
        try:
            if i_variable.lower() == "sm":
                sm_slice = sm_dummy[:,:,i].values
                vod_slice = np.full((lats,lons), const_variable_value)
                t_slice = np.full((lats,lons), t_cons)

            elif i_variable.lower() == "vod":
                sm_slice =  np.full((lats,lons), const_variable_value)
                vod_slice = vod_dummy[:,:,i].values
                t_slice = np.full((lats,lons), t_cons)

            elif i_variable.lower() == "t":
                sm_slice =  np.full((lats,lons), const_variable_value)
                vod_slice = np.full((lats, lons), vod_cons)
                t_slice = t_dummy[:,:,i].values

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
                # T_theor=THolmes.values.astype('double'),
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
                coords= get_coords()["coords"],
            ).expand_dims(i=[i])

            da = da.sel(lat=slice(bbox[3], bbox[1]),
                                lon=slice(bbox[0], bbox[2]))

            TbV_sim = da["TbV_sim"]
            TbH_sim = da["TbH_sim"]

            mpdi = ((TbV_sim - TbH_sim) / (TbV_sim + TbH_sim)).values.flatten()
            Tb_ratio = (TbH_sim / TbV_sim).values.flatten()

            X = Tb_ratio.reshape(-1, 1)
            y = mpdi

            mask = np.isfinite(X[:, 0]) & np.isfinite(y)  # removes NaN + inf
            Xf = X[mask]
            yf = y[mask]

            ransac = HuberRegressor()
            ransac.fit(Xf, yf)
            m = ransac.coef_[0]
            c = ransac.intercept_
            m = np.where(abs(m) > 0.1, m, np.nan)
            c = np.where(abs(c) > 0.1, c, np.nan)

            opt_sim_list.append(np.nanmean(opt_sim))

            tbv_list.append(TbV_sim.mean().item())
            tbh_list.append(TbH_sim.mean().item())
            m_list.append(m)
            c_list.append(c)
            print(f"succes data: {i}")

        except Exception as e:
            print(e)

    const_tbh_dict.update({const_variable_value : tbv_list})
    const_tbv_dict.update({const_variable_value : tbh_list})
    const_m_dict.update({const_variable_value : m_list})
    const_c_dict.update({const_variable_value : c_list})




##
var_dict = {"sm" :sm_i,
            "vod": vod_i,
            "t": t_i}


keys = list(const_c_dict.keys())
n_lines = len(keys)
# Using a single color map for clarity, or you can stick to your two-tone approach
colors = plt.cm.Blues(np.linspace(1, 0.3, n_lines))

plt.figure(figsize=(10, 6))

for i, key in enumerate(keys):
    # Slice the x-axis variable to match the length of the data
    x_data = var_dict[i_variable][0:len(const_tbv_dict[key])]

    plt.plot(x_data, const_tbv_dict[key], color=colors[i], linestyle='-', alpha=0.8)
    # plt.plot(x_data, const_tbh_dict[key], color=colors[i], linestyle='-', alpha=0.8)

custom_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(n_lines)]
type_lines = [Line2D([0], [0], color='gray', linestyle='-'),
              Line2D([0], [0], color='gray', linestyle='--')]

legend1 = plt.legend(custom_lines, keys, title="SM", loc='upper right')
plt.gca().add_artist(legend1)

# plt.legend(type_lines, ['TbV', 'TbH'], loc='lower right', title="pol")

plt.xlabel(i_variable)
plt.ylabel("Kelvin")
plt.title(f"TbV")
plt.ylim([90,300])
# plt.grid(F, which='both', linestyle='--', alpha=0.5)
plt.show()