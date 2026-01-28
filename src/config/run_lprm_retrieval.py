import sys
import os

# Get the absolute path of the directory containing this script (src/config)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (src)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path if it's not there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import datetime
import os
import pandas as pd
from scipy.stats import pearsonr
from readers.Sat import BTData, LPRMData
import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pandas as pd
import dask.array as da
from utilities.utils import (
    bbox,
    mpdi,
    extreme_hull_vals,
    find_common_coords,
    get_dates,
    convex_hull,
    pearson_corr,
    save_nc,
)
from utilities.retrieval_helpers import (
    soil_canopy_temperatures,
    interceptor,
    dummy_line, retrieve_LPRM,tiff_df
)
from utilities.plotting import scatter_density, plot_maps_LPRM, plot_maps_day_night, plot_timeseries
from config.paths import path_lprm, path_bt, path_aux
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from lprm.retrieval.lprm_general import load_aux_file
from lprm.retrieval.lprm_v6_1.parameters import get_lprm_parameters_for_frequency
from lprm.satellite_specs import get_specs
from utilities.run_lprm import run_band as run_band_py

overpass = "day"
sensor = "AMSR2"
# sat_band = "C1"
frequencies={
    'X': 10.7,
    'C1': 6.9,
             'C2': 7.3,
             'KU': 18.7,
             'K': 23.8,
             'Ka': 36.5
             }

start_date = "2020-01-01"
end_date = "2021-12-31"
selector = 0
theor_dict = {0 : "plain",
              1 : "edit"}
datelist = get_dates(start_date, end_date, freq = "D")
avg_dict= {}
std_dict = {}

sm_dict = {}
vod_dict = {}

def get_mpdi(v,h):
    return ((v-h)/(v+h))

for sat_band in list(frequencies.keys())[0:1]: # Set this to the index which Freq. u want
    print(sat_band)
    sm_list = []
    vod_list = []
    for d in datelist:
        print(d)
        try:
            pattern = f"{sensor}_l1bt_{overpass}_{d.strftime("%Y%m%d")}_{25}km.nc"
            bt_filepath = os.path.join(path_bt,sensor,overpass, d.strftime("%Y%m"),pattern)

            bt_data = xr.open_dataset(bt_filepath, decode_timedelta=False).isel(time = 0)
            Tbv = bt_data[f"bt_{frequencies[sat_band]}V"]
            Tbh = bt_data[f"bt_{frequencies[sat_band]}H"]

            CH = bt_data[f"bt_6.9H"]
            CV = bt_data[f"bt_6.9V"]
            KuH =  bt_data[f"bt_18.7H"]
            KuV =  bt_data[f"bt_18.7V"]
            KaV =  bt_data[f"bt_36.5V"]

            mpdi_ku = ((Tbv-Tbh)/(Tbv+Tbh))
            mpdi_ku = ((Tbv-Tbh)/(Tbv+Tbh))

            THolmes  = (0.893 * bt_data[f"bt_36.5V"]) + 44.8

            Tveg = ((0.893 * KuH) /
                     (1 - (get_mpdi(KuV,KuH) / abs(0.46)))) + 44.8
            # Tveg = xr.where(abs(Tveg- THolmes) >20,THolmes,Tveg)

            Tsoil = ((0.893 * CH) /
                     (1 - (get_mpdi(CV,CH) / abs(0.56)))) + 44.8
            # Tsoil = xr.where(abs(Tsoil- THolmes) >20,THolmes,Tsoil)

            SND = load_aux_file(0.25,"SND")
            CLY = load_aux_file(0.25,"CLY")
            BLD = load_aux_file(0.25,"BLD")

            specs = get_specs(sensor.upper())
            params = get_lprm_parameters_for_frequency(sat_band, specs.incidence_angle)
            freq = get_specs(sensor.upper()).frequencies[sat_band.upper()]

            sm, vod= par100.run_band(
                Tbv.values.astype('double'),
                Tbh.values.astype('double'),
                THolmes.values.astype('double'),
                SND,
                CLY,
                BLD,
                params.Q,
                params.w,
                0,
                specs.incidence_angle[0],
                params.h1,
                params.h2,
                params.vod_Av,
                params.vod_Bv,
                float(freq),
                params.temp_freeze,
                False,
                None,
                T_soil = Tsoil.values.astype('double'),
                T_canopy = Tveg.values.astype('double'),
                # T_theor= np.zeros_like(Tbv.values, dtype=np.float64),
                Theory_select = selector
            )

            sm  = xr.where(sm>0,sm,np.nan)
            dataset = xr.DataArray(
                data=sm,
                dims=Tbv.dims,
                coords=Tbv.coords,
                name=f'sm'
            ).to_dataset()

            vod  = np.where(vod>0,vod,np.nan)
            dataset[f'VOD_{sat_band}'] = (("lat", "lon"), vod)
            dataset[f'T_soil'] = Tsoil
            dataset[f'T_veg'] = Tveg

            # sm_list.append(dataset[f'sm'])
            vod_list.append(dataset[f'VOD_{sat_band}'] )

        except Exception as e:
            print(e)
            pass

        # sm_da = xr.concat(sm_list, dim='time')
        vod_da = xr.concat(vod_list, dim='time')

        # sm_dict.update({f"SM_{sat_band}" : sm_da})
        vod_dict.update({f"VOD_{sat_band}" : vod_da})

sm_dataset = xr.Dataset(sm_dict)
vod_dataset = xr.Dataset(vod_dict)

merged_dataset =sm_dataset.merge(vod_dataset)
comp = dict(zlib=True, complevel=5, shuffle=True, dtype='float32')
encoding = {var : comp for var in merged_dataset.var() }

merged_dataset.to_netcdf(f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
                         f"LPRM/07_debug/daytime_retrieval/LPRM_retrievals/27Jan/"
                         f"VOD_{overpass}_{theor_dict[selector]}.nc",
                         encoding=encoding)
