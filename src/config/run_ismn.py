import pandas as pd
from ismn.interface import ISMN_Interface
from ismn.meta import Depth
import xarray as xr
import numpy as np
from pandas import Timestamp,Timedelta
from utilities.plotting import temp_sm_plot
from config.paths import sat_stack_path,ismn_data_path, path_aux
from utilities.retrieval_helpers import retrieve_LPRM,tiff_df
from utilities.utils import local_solar_time,get_dates
import itertools
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

def run_ismn_multi_site(satellite_data,
                        ISMN_instance,
                        site_list,
                        ts_cutoff,
                        depth_selection,
                        network = "SCAN"
                        ):
    """
    Plot Soil T and Soil Moisture in loops for the selected ISMN sites.
    :param satellite_data: Satellite data containing LPRM derived values. Usually the output of "run_triangle.py"
    :param ISMN_instance: ISMN_Interface instance from .zip
    :param site_list: List of sites to process from network.
    :param ts_cutoff: Max date to limit sensor timespan.
    :param depth_selection: How deep does the sensor go?
    :param network: Which ISMN network do you use? Default: SCAN
    """

    NETWORK = ISMN_instance[network]
    for i in site_list:

        STATION = NETWORK[i]
        print(i)

        try:
            for  _, _sensor_sm in NETWORK.iter_sensors(variable='soil_moisture',
                                                             depth=depth_selection,
                                                             filter_meta_dict={
                                                                 'station': [i],
                                                             }):

                if _sensor_sm.metadata["timerange_from"][1] > ts_cutoff:
                    ismn_sm = _sensor_sm.read_data()

                for _, _sensor_t in NETWORK.iter_sensors(variable='soil_temperature',
                                                               depth=depth_selection,
                                                               filter_meta_dict={
                                                                   'station': [i],
                                                               }):
                    if _sensor_t.metadata["timerange_from"][1] > ts_cutoff:

                        ismn_t = _sensor_t.read_data()



            data =  satellite_data.sel(
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


##
sat_data = xr.open_dataset(sat_stack_path)
timespan = sat_data.time.values
ISMN_stack = ISMN_Interface(ismn_data_path, parallel=True)
NETWORK_stack = ISMN_stack["SCAN"]
ts_cutoff = Timestamp("2024-06-01")
depth_selection = Depth(0., 0.1)
site_list = [ 'Lind#1']

station_user = 'AlabamaHills'
SINGLE_STATION = NETWORK_stack[station_user]

_sat_data = sat_data.sel(
            LAT =SINGLE_STATION.lat,
            LON =SINGLE_STATION.lon,
            method = "nearest"
        ).expand_dims(['LAT','LON']).to_dataframe()

base_coniditons = (
    (ISMN_stack.metadata['instrument'].depth_to < 0.1) &
    (ISMN_stack.metadata['instrument'].depth_from >= 0) &
    (ISMN_stack.metadata["timerange_to"].val > ts_cutoff + Timedelta(days=1)) &
    (ISMN_stack.metadata['station'].val == station_user)
)

conditions_sm = base_coniditons & (ISMN_stack.metadata['variable'].val == 'soil_moisture')
conditions_st = base_coniditons & (ISMN_stack.metadata['variable'].val == 'soil_temperature')

ids_sm = ISMN_stack.metadata[conditions_sm].index.to_list()
ids_st = ISMN_stack.metadata[conditions_st].index.to_list()
ts_sm, meta_sm = ISMN_stack.read(ids_sm, return_meta=True)
ts_st, meta_st = ISMN_stack.read(ids_st, return_meta=True)
# ts_sm.plot(label = "in situ")
# _sat_data["SM_X"].reset_index(level=["LAT", "LON"],drop=True).plot(label = "LPRM_X")
# t = _sat_data.index.get_level_values("time")
# plt.xlim(timespan.min(), timespan.max())
# plt.legend()
# plt.show()
aux_df = tiff_df(path_aux)
T_soil_range = np.arange(273,330,2)
T_canopy_range = np.arange(273,330,2)
iterables = [T_soil_range,T_canopy_range]
dates = get_dates(Timestamp("2024-01-01"), Timestamp("2024-12-01"), freq = "ME")

for day in dates:

    print(day)
    try:
        sat_day = _sat_data.drop(columns = ["SM_ADJ"]).xs(day, level="time")

        sol_time = local_solar_time(sat_day["SCANTIME_BT"].values.item(),
                    day,
                    sat_day.index.get_level_values("LON")[0])

        i = ts_sm.index.get_indexer([day], method="nearest")[0]
        closest_row = ts_sm.iloc[i]
        SM_target = closest_row.xs("soil_moisture", level="variable").dropna().values[0]

        logger = {
            "Soil" : [],
            "Canopy" : [],
            "SM" : [],
            "dif" : []
        }

        for T_soil_i,T_canopy_i in itertools.product(*iterables):

            lprm_day = retrieve_LPRM(sat_day,
                                     aux_df,
                                     "AMSR2",
                                     "X",
                                     T_soil_test=T_soil_i,
                                     T_canopy_test=T_canopy_i,
                                     ).to_dataframe()

            SM_i = lprm_day["SM_ADJ"].values.item()
            logger["Soil"].append(T_soil_i)
            logger["Canopy"].append(T_canopy_i)
            logger["SM"].append(SM_i)
            logger["dif"].append(SM_target - SM_i)

        df_logger = pd.DataFrame(logger).sort_values(by=['dif'],key=abs)
        df = df_logger.copy()
        plt.figure(figsize=(8, 6))

        vmin=-0.2
        vcenter=0
        vmax=0.2
        norm = TwoSlopeNorm(vmin=vmin,
                            vcenter=vcenter,
                            vmax=vmax)

        sc = plt.scatter(
            df_logger['Soil'],
            df_logger['Canopy'],
            c=df_logger['dif'],
            cmap='bwr',
            norm=norm,
            edgecolor='k',
            s=80
        )

        # create colorbar
        cbar = plt.colorbar(sc, ticks=np.linspace(vmin, vmax, 5))
        cbar.set_label('dif')
        plt.scatter(lprm_day["TSURF"],lprm_day["TSURF"],color='gold')
        plt.xlabel('Soil')
        plt.ylabel('Canopy')
        plt.title(f"{SM_target}\n {day}" )
        plt.grid(False)
        plt.show()
    except ValueError as e:
        print(e)
        continue

# if __name__ == "__main__":
#     run_ismn_multi_site(satellite_data=sat_data,
#                         ISMN_instance=ISMN_stack,
#                         site_list=site_list,
#                         ts_cutoff=ts_cutoff,
#                         depth_selection=depth_selection)