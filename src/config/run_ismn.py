from ismn.interface import ISMN_Interface
from ismn.meta import Depth
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pandas import Timestamp
from utilities.plotting import temp_sm_plot
sat_sm = xr.open_dataset("/home/ddkovacs/Desktop/personal/daytime_retrievals/datasets/US_2024.nc")


ISMN_stack = ISMN_Interface('/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/debug/daytime_retrieval/ismn_data/SCAN.zip',
                    parallel=True)
NETWORK_stack = ISMN_stack["SCAN"]


station_user = "Abrams"
ts_cutoff = Timestamp("2015-01-01")
depth_selection = Depth(0., 0.1)


stat_list = [ 'Lind#1', 'Lindsay', 'LittleRedFox', 'LittleRiver', 'Livingston-UWA', 'LosLunasPMC', 'LovellSummit', 'LovelockNNR', 'LowerMulchatna', 'LyeBrook', 'MahantangoCk', 'MammothCave', 'ManaHouse', 'Mandan#1', 'Manderfield', 'MarbleCreek', 'MaricaoForest', 'MarkTwainHS', 'MascomaRiver', 'Mason#1', 'Mayday', 'McAllisterFarm',]

for i in stat_list:
    STATION = NETWORK_stack[i]
    print(i)
    try:
        for  _, _sensor_sm in NETWORK_stack.iter_sensors(variable='soil_moisture',
                                                         depth=depth_selection,
                                                         filter_meta_dict={
                                                             'station': [i],
                                                         }):

            if _sensor_sm.metadata["timerange_from"][1] > ts_cutoff:
                ismn_sm = _sensor_sm.read_data()

            for _, _sensor_t in NETWORK_stack.iter_sensors(variable='soil_temperature',
                                                           depth=depth_selection,
                                                           filter_meta_dict={
                                                               'station': [i],
                                                           }):
                if _sensor_t.metadata["timerange_from"][1] > ts_cutoff:

                    ismn_t = _sensor_t.read_data()



        data =  sat_sm.sel(
            LAT =STATION.lat,
            LON =STATION.lon,
            method = "nearest"
        )

        sm_adj = data["SM_ADJ"]
        sm_x = data["SM_X"]
        sat_t_soil = data["T_soil_hull"]-273.15
        sat_t_canopy = data["T_canopy_hull"] -273.15
        sat_t = data["TSURF"] -273.15

        temp_sm_plot(
            ismn_t,
            sat_t,
            sat_t_soil,
            sat_t_canopy,
            ismn_sm,
            sm_x,
            sm_adj,
            **{
            "name" : STATION.name,
            "lat" : np.round(STATION.lat,2),
            "lon" : np.round(STATION.lon,2),
        }
        )
    except:
        continue