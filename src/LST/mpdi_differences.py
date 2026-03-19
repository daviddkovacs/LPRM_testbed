from datacube_loader import MICROWAVE_datacube
from datacube_utilities import mpdi, calc_Holmes_temp, frequencies, crop2roi, ravel_roi_time
import pandas as pd
import matplotlib.pyplot as plt
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from lprm.retrieval.lprm_general import load_aux_file
from lprm.retrieval.lprm_v6_1.parameters import (
    get_lprm_parameters_for_frequency,
)
import xarray as xr
import numpy as np
from plot_functions import plot_hexbin

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

def retrieve_LPRM(TB_DATASET, HOLMES_T, band, SM_input = None, VOD_input = None):
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
    lprm_list_tsim = []

    for t in times:
        print(t.dt.date.item())
        tb_map = TB_DATASET.sel(time = t).compute()
        holmes_t = HOLMES_T.sel(time = t).compute()

        if SM_input is not None:
            sm_input = SM_input.sel(time = t).compute().values
            vod_input = VOD_input.sel(time = t).compute().values
        else:
            sm_input = None
            vod_input = None
        aux_data_dict = {
            "sand": load_aux_file(0.25, "SND"),
            "clay": load_aux_file(0.25, "CLY"),
            "bld": load_aux_file(0.25, "BLD"),
        }
        params = get_lprm_parameters_for_frequency(band, inc_angle)

        sm, vod,tsim = par100.run_band(
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
            SM_map_night = sm_input,
            VOD_map_night = vod_input,
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

        if SM_input is not None:
            tsim_da = xr.DataArray(
                data=tsim,
                coords=tb_map.coords,
                dims=tb_map.dims,
                name="tsim"
            )
            tsim_da = tsim_da.where(tsim_da>=0)
            lprm_list_tsim.append(tsim_da)

        lprm_list_sm.append(sm_da)
        lprm_list_vod.append(vod_da)

    SM_dataset = xr.concat(lprm_list_sm, dim = "time")
    VOD_dataset = xr.concat(lprm_list_vod, dim = "time")
    if SM_input is not None:
        TSIM_dataset = xr.concat(lprm_list_tsim, dim="time")
    else:
        TSIM_dataset = np.zeros(9)

    return SM_dataset, VOD_dataset, TSIM_dataset


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
    SM_NIGHT, VOD_NIGHT,_ = retrieve_LPRM(TB_DATASET=AMSR2_NIGHT, HOLMES_T=HOLMES_T_NIGHT, band=band_current)
    # Highly experimental! Dummy variables are given for TB and HOLMES. TSIM is obtained by
    # running LPRM in reverse.
    _, _, TSIM_DAY = retrieve_LPRM(TB_DATASET=AMSR2_NIGHT, HOLMES_T=HOLMES_T_NIGHT, band=band_current,
                               SM_input=SM_NIGHT, VOD_input=VOD_NIGHT)
##
    threshold = 0.000005
    mpdi_delta_band = MPDI_deltas[band_current]

    low_mpdi_mask = xr.where((mpdi_delta_band >= -threshold) & (mpdi_delta_band <= threshold),
                             1,0).compute()

    SM_low_mpdi = xr.where((low_mpdi_mask==1),SM_NIGHT,np.nan)
    VOD_low_mpdi = xr.where((low_mpdi_mask==1),VOD_NIGHT,np.nan)
    HOLMES_T_DAY_low_mpdi = xr.where((low_mpdi_mask==1),HOLMES_T_DAY,np.nan)
    AMSR2_DAY_low_mpdi = xr.where((low_mpdi_mask==1),AMSR2_DAY,np.nan)
    TSIM_low_mpdi = xr.where((low_mpdi_mask==1),TSIM_DAY,np.nan)


    ##

    roi =  [
   -180,-90,180,90
  ]

    T_KA = AMSR2_DAY_low_mpdi["bt_36.5V"]
    DELTA_T = TSIM_low_mpdi - T_KA

    F = (AMSR2_DAY_low_mpdi[f"bt_{frequencies["ku".upper()]}H"]
         /AMSR2_DAY_low_mpdi[f"bt_{frequencies["ka".upper()]}V"])

    date_range = pd.date_range(start="2018-01-01",end="2019-01-01",freq="MS")

    for i in date_range.month:
        month_selector = (DELTA_T.time.dt.month == i)
        df = pd.DataFrame({
            "DELTA_T": ravel_roi_time(DELTA_T,roi,month_selector,method="nearest"),
            "F": ravel_roi_time(F,roi,month_selector,method="nearest"),
            "T_KA": ravel_roi_time(T_KA,roi,month_selector,method="nearest"),
            "TSIM_low_mpdi" : ravel_roi_time(TSIM_low_mpdi,roi,month_selector,method="nearest"),
            "VOD_low_mpdi": ravel_roi_time(VOD_low_mpdi,roi,month_selector,method="nearest"),
            "SM_low_mpdi": ravel_roi_time(SM_low_mpdi,roi,month_selector,method="nearest"),
        })

        plot_hexbin(df,
                    "TSIM_low_mpdi",
                    "F",
                    color_of_points="VOD_low_mpdi",
                    xlim=[None, None], ylim=[None, None],
                    # cbar_min= 0.95, cbar_max= 1.05,
                    title_string=f"{i}")