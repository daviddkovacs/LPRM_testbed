from datacube_loader import MICROWAVE_datacube
from datacube_utilities import mpdi, calc_Holmes_temp, frequencies
import pandas as pd
import matplotlib.pyplot as plt
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from lprm.retrieval.lprm_general import load_aux_file
from lprm.retrieval.lprm_v6_1.parameters import (
    get_lprm_parameters_for_frequency,
)


def load_AMSR2_daily(bbox,time_start,time_stop):
    """
    Load day/night AMSR2 TBs. we need to re-assign the time dimension, as MICROWAVE_datacube assigned the average scantime
    values within bbox (skews observation times when bbox is global)
    :param bbox: List[min_lon,min_lat,max_lon,max_lat]
    :param time_start: date
    :param time_stop: date
    :return: xr Dataset of day and night TBs with daily timestamps
    """

    AMSR2_DAY = MICROWAVE_datacube(bbox=bbox,
                                  overpass="day",
                                  time_start=time_start,
                                  time_stop=time_stop)

    AMSR2_NIGHT = MICROWAVE_datacube(bbox=bbox,
                                  overpass="night",
                                  time_start=time_start,
                                  time_stop=time_stop)


    AMSR2_DAY['time'] = pd.to_datetime(AMSR2_DAY.time.dt.date.values)
    AMSR2_NIGHT['time'] = pd.to_datetime(AMSR2_NIGHT.time.dt.date.values)

    return AMSR2_DAY, AMSR2_NIGHT


def calc_MPDI_bands(AMSR2_DAY,AMSR2_NIGHT, list_of_bands=["c2", "x", "ku"]):
    """
    We calculate MPDIs for different frequencies
    :param AMSR2_DAY: Daytime TB stack
    :param AMSR2_NIGHT: Nighttime TB stack
    :return: Dictionary with keys as bands and values as MPDI datasets
    """

    MPDI_DAY_dict = {}
    MPDI_NIGHT_dict = {}

    for band in list_of_bands:
        MPDI_DAY_dict[band] = mpdi(AMSR2_DAY,band)
        MPDI_NIGHT_dict[band] = mpdi(AMSR2_NIGHT,band)

    return MPDI_DAY_dict, MPDI_NIGHT_dict


def calc_MPDI_difference(MPDI_day, MPDI_night, list_of_bands=["c2", "x", "ku"]):
    """
    We calculate the difference in MPDI. Night-Day!!!
    :param MPDI_day: MPDI calculated for daytime obs
    :param MPDI_night: MPDI calculated for nighttime obs
    :param list_of_bands: frequencies needed to calc MPDI dif for
    :return: dictionary containing list_of_bands MPDI differences
    """

    MPDI_difference_dict = {}

    for band in list_of_bands:
        MPDI_difference_dict[band] = MPDI_night[band] - MPDI_day[band]
    return MPDI_difference_dict


##
if __name__=="__main__":

    bbox = [-180, -90, 180, 90]
    time_start = "2018-01-01"
    time_stop = "2019-01-01"
    bandlist = ["c2", "x", "ku"]

    AMSR2_DAY, AMSR2_NIGHT = load_AMSR2_daily(bbox = bbox,time_start=time_start,time_stop=time_stop)
    HOLMES_T_NIGHT, HOLMES_T_DAY = calc_Holmes_temp(AMSR2_NIGHT), calc_Holmes_temp(AMSR2_DAY)
    MPDI_DAY , MPDI_NIGHT = calc_MPDI_bands(AMSR2_DAY=AMSR2_DAY,AMSR2_NIGHT=AMSR2_NIGHT, list_of_bands=bandlist)
    MPDI_deltas = calc_MPDI_difference(MPDI_day=MPDI_DAY, MPDI_night=MPDI_NIGHT, list_of_bands=bandlist)

##

    times = AMSR2_NIGHT.time
    LPRM_dict = {}
    inc_angle = 55.0

    for band in bandlist:
        lprm_list = []
        band = band.upper()
        freq = frequencies[band.upper()]

        for day in times:
            print(day.dt.date.item())
            tb_map = AMSR2_NIGHT.sel(time = day).compute()
            holmes_t = HOLMES_T_NIGHT.sel(time = day).compute()

            aux_data_dict = {
                "sand": load_aux_file(0.25, "SND"),
                "clay": load_aux_file(0.25, "CLY"),
                "bld": load_aux_file(0.25, "BLD"),
            }
            params = get_lprm_parameters_for_frequency(band, inc_angle)

            sm, vod = par100.run_band(
                tb_map[f"bt_{freq}V"].values,
                tb_map[f"bt_{freq}H"].values,
                holmes_t.values,
                aux_data_dict["sand"],
                aux_data_dict["clay"],
                aux_data_dict["bld"],
                params.Q,
                params.w,
                params.opt_atm,
                inc_angle,
                params.h1,
                params.h2,
                params.vod_Av,
                params.vod_Bv,
                float(freq),
                params.temp_freeze,
                False,    # apply VOD correction if mean is passed
                None,                # pass mean VOD of backwards window
            )


    # threshold = 0.005
    #
    # mpdi_current = MPDI_deltas["x"].isel(time=100).compute()
    # mpdi_filtered = mpdi_current.where((mpdi_current >= -threshold) & (mpdi_current <= mpdi_current))
    #
    # plt.figure(figsize=(20,12))
    # mpdi_filtered.plot(vmin= -threshold, vmax = threshold, cmap = "coolwarm")
    # plt.show()