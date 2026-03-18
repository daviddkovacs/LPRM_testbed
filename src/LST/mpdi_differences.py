from datacube_loader import MICROWAVE_datacube
from datacube_utilities import mpdi
import pandas as pd
import matplotlib.pyplot as plt

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
    MPDI_DAY , MPDI_NIGHT = calc_MPDI_bands(AMSR2_DAY=AMSR2_DAY,AMSR2_NIGHT=AMSR2_NIGHT, list_of_bands=bandlist)
    MPDI_deltas =  calc_MPDI_difference(MPDI_day=MPDI_DAY, MPDI_night=MPDI_NIGHT, list_of_bands=bandlist)
##
    t= 20
    test_band = "x"
    test_day = MPDI_DAY[test_band].isel(time = t).compute()
    test_night = MPDI_NIGHT[test_band].isel(time = t).compute()
    test_dif = test_night - test_day


    plt.figure(figsize=(20,12))
    test_dif.plot(vmin = -0.01, vmax = 0.01, cmap = "coolwarm")
    plt.title(f"MPDI{test_band} difference (night - day)")
    plt.show()

    ##
    dates = pd.date_range(start="2018-01-01", end="2019-01-01", freq="D")


    for i in range(len(dates) - 1):

        time_start = dates[i].strftime("%Y-%m-%d")
        time_stop = dates[i + 1].strftime("%Y-%m-%d")

        month_mean_day = MPDI_DAY[test_band].sel(time = slice(time_start,time_stop))
        month_mean_night = MPDI_NIGHT[test_band].sel(time = slice(time_start,time_stop))

        month_diff = (month_mean_night - month_mean_day).compute()
        mean_diff  = month_diff.mean(dim = "time")

        minval = -0.0001
        maxval = 0.0001
        filtered_diff = mean_diff.where((mean_diff >= minval) & (mean_diff <= maxval))

        plt.figure(figsize=(20,12))
        filtered_diff.plot(vmin = minval, vmax = maxval, cmap = "coolwarm")
        plt.title(f"MPDI{test_band} {time_start} dif (night - day)")
        plt.show()