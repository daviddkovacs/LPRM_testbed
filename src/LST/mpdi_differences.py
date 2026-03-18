from datacube_loader import MICROWAVE_datacube
from datacube_utilities import mpdi, calc_Holmes_temp, frequencies
import pandas as pd
import matplotlib.pyplot as plt
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from lprm.retrieval.lprm_general import load_aux_file
from lprm.retrieval.lprm_v6_1.parameters import (
    get_lprm_parameters_for_frequency,
)
import xarray as xr
import numpy as np

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

def retrieve_LPRM(TB_DATASET, HOLMES_T, band):
    """
    Retrieve LPRM, traditional method. Input is Brightness temps, Holmes "KA" temp and band
    :return: SM and VOD datasets
    """
    times = TB_DATASET.time
    inc_angle = 55.0

    band = band.upper()
    freq = frequencies[band.upper()]

    lprm_list_sm = []
    lprm_list_vod = []
    for t in times:
        print(t.dt.date.item())
        tb_map = TB_DATASET.sel(time = t).compute()
        holmes_t = HOLMES_T.sel(time = t).compute()

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
            False,
            None,
        )

        sm_da = xr.DataArray(
            data=sm,
            coords=tb_map.coords,
            dims=tb_map.dims,
            name="sm"
        )

        vod_da = xr.DataArray(
            data=vod,
            coords=tb_map.coords,
            dims=tb_map.dims,
            name="vod"
        )
        sm_da = sm_da.where(sm_da>=0)
        vod_da = vod_da.where(vod_da>=0)

        lprm_list_sm.append(sm_da)
        lprm_list_vod.append(vod_da)

    SM_dataset = xr.concat(lprm_list_sm, dim = "time")
    VOD_dataset = xr.concat(lprm_list_vod, dim = "time")

    return SM_dataset, VOD_dataset


def threshold_by_mpdi(SM,VOD, MPDI, threshold):
    """
    Where MPDI difference (night-day) is between +- threshold, we mask SM and VOD retrievals
    :param SM: SM dataset
    :param VOD: VOD dataset
    :param MPDI: MPDI difference at band
    :param threshold: usually a low float
    :return: masked SM, masked VOD
    """
    low_mpdi_mask = xr.where((MPDI >= -threshold) & (MPDI <= threshold),1,0).compute()

    SM_low_mpdi = xr.where((low_mpdi_mask==1),SM,np.nan)
    VOD_low_mpdi = xr.where((low_mpdi_mask==1),VOD,np.nan)

    return SM_low_mpdi, VOD_low_mpdi

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
    band_current = "x"
    SM, VOD = retrieve_LPRM(TB_DATASET=AMSR2_NIGHT, HOLMES_T=HOLMES_T_NIGHT, band=band_current)

##
    threshold = 0.0005
    mpdi_delta_band = MPDI_deltas[band_current]

    SM_low_mpdi, VOD_low_mpdi = threshold_by_mpdi(SM=SM, VOD=VOD,MPDI=mpdi_delta_band, threshold=threshold)
