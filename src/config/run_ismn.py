from tokenize import String

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
                        sites,
                        ts_cutoff,
                        depth_selection,
                        network = "SCAN"
                        ):
    """
    Plot Soil T and Soil Moisture in loops for the selected ISMN sites.
    :param satellite_data: Satellite data containing LPRM derived values. Usually the output of "run_triangle.py"
    :param ISMN_instance: ISMN_Interface instance from .zip
    :param sites: List of sites to process from network.
    :param ts_cutoff: Max date to limit sensor timespan.
    :param depth_selection: How deep does the sensor go? Dict : {"start": float, "end": float}
    :param network: Which ISMN network do you use? Default: "SCAN"
    """

    NETWORK = ISMN_instance[network]
    for i in sites:

        STATION = NETWORK[i]
        print(i)

        # try:
        for  _, _sensor_sm in NETWORK.iter_sensors(variable='soil_moisture',
                                                         depth=Depth(depth_selection["start"],depth_selection["end"]),
                                                         filter_meta_dict={
                                                             'station': [i],
                                                         }):

            if _sensor_sm.metadata["timerange_to"][1] > ts_cutoff:
                ismn_sm = _sensor_sm.read_data()

            for _, _sensor_t in NETWORK.iter_sensors(variable='soil_temperature',
                                                           depth=Depth(depth_selection["start"],depth_selection["end"]),
                                                           filter_meta_dict={
                                                               'station': [i],
                                                           }):
                if _sensor_t.metadata["timerange_to"][1] > ts_cutoff:

                    ismn_t = _sensor_t.read_data()

        data =  satellite_data.sel(
            LAT =STATION.lat,
            LON =STATION.lon,
            method = "nearest"
        )

        sm_adj = data["SM_ADJ"]
        sm_x = data["SM_X"]
        sat_t_soil = data["T_soil_hull"]
        sat_t_canopy = data["T_canopy_hull"]
        sat_t = data["TSURF"]

        temp_sm_plot(
            ismn_t ,
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
        # except Exception as e:
        #     print(e)
        #     continue


##

def temperature_distribution(satellite_data,
                        ISMN_instance,
                        site,
                        ts_cutoff,
                        depth_selection,
                        dates,
                        network = "SCAN"
                        ):
    """
    This function runs LPRM for a number of combinations of T_canopy and T_soil. It plots the difference to target SM.
    :param satellite_data: Satellite data containing BT  values. Usually the output of "run_triangle.py"
    :param ISMN_instance: ISMN_Interface instance from .zip
    :param site: List of site (singular) to process from network.
    :param ts_cutoff: Max date to limit sensor timespan.
    :param depth_selection: How deep does the sensor go? Dict : {"start": float, "end": float}
    :param dates: Timestamps of the T distribution to plot
    :param network:  Which ISMN network do you use? Default: "SCAN"
    """

    NETWORK_stack = ISMN_instance[network]
    SINGLE_STATION = NETWORK_stack[site]

    _sat_data = satellite_data.sel(
                LAT =SINGLE_STATION.lat,
                LON =SINGLE_STATION.lon,
                method = "nearest"
            ).expand_dims(['LAT','LON']).to_dataframe()

    base_coniditons = (
        (ISMN_stack.metadata['instrument'].depth_from >= depth_selection["start"]) &
        (ISMN_stack.metadata['instrument'].depth_to < depth_selection["end"]) &
        (ISMN_stack.metadata["timerange_to"].val > ts_cutoff + Timedelta(days=1)) &
        (ISMN_stack.metadata['station'].val == station_user)
    )

    conditions_sm = base_coniditons & (ISMN_stack.metadata['variable'].val == 'soil_moisture')
    conditions_st = base_coniditons & (ISMN_stack.metadata['variable'].val == 'soil_temperature')

    ids_sm = ISMN_stack.metadata[conditions_sm].index.to_list()
    ids_st = ISMN_stack.metadata[conditions_st].index.to_list()
    ts_sm, meta_sm = ISMN_stack.read(ids_sm, return_meta=True)
    ts_st, meta_st = ISMN_stack.read(ids_st, return_meta=True)

    aux_df = tiff_df(path_aux)
    T_soil_range = np.arange(273,330,1)
    T_canopy_range = np.arange(273,330,1)
    iterables = [T_soil_range,T_canopy_range]

    n = len(dates)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 4*nrows),
        constrained_layout=True
    )
    axes = axes.flatten()

    vmin = -0.2
    vcenter = 0
    vmax = 0.2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    last_scatter = None

    failed_switch = False
    for ax, day in zip(axes, dates):

        print(failed_switch)
        day_i = day + Timedelta(days=1) if failed_switch else day
        print(day)

        try:
            sat_day = _sat_data.drop(columns=["SM_ADJ"]).xs(day_i, level="time")

            sol_time = local_solar_time(
                sat_day["SCANTIME_BT"].values.item(),
                day_i,
                sat_day.index.get_level_values("LON")[0]
            )

            i_sm = ts_sm.index.get_indexer([sol_time], method="nearest")[0]
            i_st = ts_st.index.get_indexer([sol_time], method="nearest")[0]
            closest_insitu_sm = ts_sm.iloc[i_sm]
            closest_insitu_st = ts_st.iloc[i_st]

            SM_target = closest_insitu_sm.xs("soil_moisture", level="variable").dropna().values[0]
            ST_target = closest_insitu_st.xs("soil_temperature", level="variable").dropna().values[0]+273.15

            logger = {"Soil": [], "Canopy": [], "SM": [], "dif": []}

            for T_soil_i, T_canopy_i in itertools.product(*iterables):

                lprm_day = retrieve_LPRM(
                    sat_day,
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

            df_logger = pd.DataFrame(logger).sort_values(by="dif", key=abs)

            sc = ax.scatter(
                df_logger['Soil'],
                df_logger['Canopy'],
                c=df_logger['dif'],
                cmap='bwr',
                norm=norm,
                edgecolor='k',
                s=60,
            )
            last_scatter = sc

            ax.scatter(lprm_day["TSURF"], lprm_day["TSURF"], color='gold', label = "T_eff LPRM")
            ax.scatter(ST_target, 273, color='green', label = "T eff True")

            ax.set_title(f"{SM_target} | {closest_insitu_sm.name}")
            ax.set_xlabel("Soil")
            ax.set_ylabel("Canopy")
            ax.grid(False)
            failed_switch = False

        except Exception as e:
            print(e)
            # ax.set_title(f"{day_i}\n{e}")
            # ax.axis("off")
            failed_switch = True
            continue

    fig.suptitle(station_user, fontsize=20, y=1.02)

    fig.colorbar(
        last_scatter,
        ax=axes,
        location="right",
        shrink=0.85,
        label="dif"
    )

    plt.show()

##

sat_data = xr.open_dataset(sat_stack_path)
ISMN_stack = ISMN_Interface(ismn_data_path, parallel=True)
ts_cutoff = Timestamp("2024-06-01")
dates = get_dates(Timestamp("2024-01-01"), Timestamp("2024-12-01"), freq="ME")

depth_selection = {"start": 0,
                   "end": 0.1}

station_user = 'Buckhorn'

if __name__ == "__main__":


    run_ismn_multi_site(satellite_data=sat_data,
                        ISMN_instance=ISMN_stack,
                        sites=  [station_user],
                        ts_cutoff=ts_cutoff,
                        depth_selection=depth_selection)


    temperature_distribution(satellite_data=sat_data,
                        ISMN_instance=ISMN_stack,
                        site=  station_user,
                        ts_cutoff=ts_cutoff,
                        depth_selection=depth_selection,
                        dates = dates)