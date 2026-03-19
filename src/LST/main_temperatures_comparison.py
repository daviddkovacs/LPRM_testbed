import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
from typing import Literal
from LST.datacube_utilities import match_MYD09_to_MYD11
from LST.plot_functions import plot_hexbin
from LST.datacube_class import DATA_READER
from datacube_utilities import (morning_evening_passes, coarsen_highres,
                                common_observations,
                                mpdi,
                                calc_Holmes_temp,
                                MW_fraction,
                                threshold_ndvi,
                                landcover_bbox_lut)
import matplotlib.pyplot as plt

def monthly_boxplots(dates,landcover,time_of_day,x_axis_scatter, y_axis_scatter, ylim=[None,None] ):
    data_monthly = {}

    for i in range(len(dates) - 1):
        time_start = dates[i].strftime("%Y-%m-%d")
        time_stop = dates[i + 1].strftime("%Y-%m-%d")

        print(time_start)
        Data = DATA_READER(region="midwest",
                           bbox=landcover_bbox_lut[landcover],
                           time_start=time_start,
                           time_stop=time_stop)

        AMSR2_data = Data.AMSR2_BT
        # MODIS_NDVI_cropped, MODIS_LST_cropped = Data.match_AMSR2_extent()

        # data_df = main_processor(MODIS_LST=MODIS_LST_cropped, MODIS_NDVI=MODIS_NDVI_cropped, AMSR2=AMSR2_data, time_of_day=time_of_day, mpdi_band=mpdi_band)
        data_df = ASMR2_arrays(AMSR2_data, time_of_day=time_of_day, mpdi_band=mpdi_band)
        data_monthly[time_start] = data_df

    all_months_data = []

    for month_key, df in data_monthly.items():
        temp_df = df[[f"{x_axis_scatter}morning", f"{y_axis_scatter}evening"]].copy()
        temp_df['Month'] = month_key
        all_months_data.append(temp_df)

    combined_df = pd.concat(all_months_data, ignore_index=True)

    melted_df = combined_df.melt(
        id_vars='Month',
        value_vars=[f"{x_axis_scatter}morning", f"{y_axis_scatter}evening"],
        var_name='pol',
        value_name='TB'
    )

    plt.figure(figsize=(16, 8))
    plt.ylim(ylim)
    sns.boxplot(
        data=melted_df,
        x='Month',
        y='TB',
        hue='pol',
        palette='muted',
        showfliers=False
    )

    plt.title(f'Monthly {x_axis_scatter} and {y_axis_scatter} ({landcover})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('TB', fontsize=12)
    # plt.ylim([245, 310])
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.show()


def monthly_scatter(dates, landcover, time_of_day, x_axis_scatter, y_axis_scatter,
                    numerator_Hpol=None, denominator_Vpol=None,
                    xlim=[250, 330], ylim=[250, 330]):

    fig, axes = plt.subplots(3, 4, figsize=(22, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(len(dates) - 1):
        time_start = dates[i].strftime("%Y-%m-%d")
        time_stop = dates[i + 1].strftime("%Y-%m-%d")

        print(time_start)
        Data = DATA_READER(region="midwest",
                           bbox=landcover_bbox_lut[landcover],
                           time_start=time_start,
                           time_stop=time_stop)

        AMSR2_data = Data.AMSR2_BT
        # MODIS_NDVI_cropped, MODIS_LST_cropped = Data.match_AMSR2_extent()

        # data_df = main_processor(MODIS_LST=MODIS_LST_cropped, MODIS_NDVI=MODIS_NDVI_cropped, AMSR2=AMSR2_data, time_of_day=time_of_day, mpdi_band=mpdi_band)
        data_df = ASMR2_arrays(AMSR2_data, time_of_day=time_of_day, mpdi_band=mpdi_band,
                               numerator_Hpol=numerator_Hpol,denominator_Vpol=denominator_Vpol)

        hb = plot_hexbin(data_df, f"{x_axis_scatter}evening", f"{y_axis_scatter}morning", xlim=xlim, ylim=ylim,
                         utc_timeofday=time_of_day, title_string=f"{landcover}\n{time_start}", ax=axes[i],
                         show_colorbar=False)
        axes[i].label_outer()

    fig.subplots_adjust(hspace=0.4, wspace=0.1, right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(hb, cax=cbar_ax)
    cbar.set_label('Count', fontsize=14)

    plt.show(block=True)
    print(landcover)


def ASMR2_arrays(AMSR2_datacube_daynight,
                 time_of_day,
                 mpdi_band = None,
                 numerator_Hpol = "ku",
                 denominator_Vpol = "ka"):
    """
    Get AMSR2 observations
    :return: dataframe with flattened arrays ready to plot in scatterplot
    """
    AMSR2_datacube = morning_evening_passes(AMSR2_datacube_daynight, time_of_day=time_of_day)
    AMSR2_datacube_mor = morning_evening_passes(AMSR2_datacube_daynight, time_of_day="morning")
    AMSR2_datacube_eve = morning_evening_passes(AMSR2_datacube_daynight, time_of_day="evening")

    AMSR2_MPDI = mpdi(AMSR2_datacube, band=mpdi_band)
    AMSR2_MPDI_mor = mpdi(AMSR2_datacube_mor, band=mpdi_band)
    AMSR2_MPDI_eve = mpdi(AMSR2_datacube_eve, band=mpdi_band)
    AMSR_LST = calc_Holmes_temp(AMSR2_datacube)

    AMSR_C1v = AMSR2_datacube["bt_6.9V"]
    AMSR_C1h = AMSR2_datacube["bt_6.9H"]
    AMSR_C2v = AMSR2_datacube["bt_7.3V"]
    AMSR_C2h = AMSR2_datacube["bt_7.3H"]
    AMSR_Xv = AMSR2_datacube["bt_10.7V"]
    AMSR_Xh = AMSR2_datacube["bt_10.7H"]
    AMSR_KUv = AMSR2_datacube["bt_18.7V"]
    AMSR_KUh = AMSR2_datacube["bt_18.7H"]
    AMSR_Kv = AMSR2_datacube["bt_23.8V"]
    AMSR_Kh = AMSR2_datacube["bt_23.8H"]
    AMSR_KAv = AMSR2_datacube["bt_36.5V"]
    AMSR_KAh = AMSR2_datacube["bt_36.5H"]
    AMSR_Wv = AMSR2_datacube["bt_89.0V"]
    AMSR_Wh = AMSR2_datacube["bt_89.0H"]

    fraction = MW_fraction(AMSR2_datacube,num=numerator_Hpol,denom=denominator_Vpol)

    _mpdi_eve = AMSR2_MPDI_eve.values.ravel()
    _mpdi_mor = AMSR2_MPDI_mor.values.ravel()

    data_df = pd.DataFrame({
        # f"AMSR2_LST_{time_of_day}": AMSR_LST.values.ravel(),
        # f"AMSR2_MPDI_{mpdi_band}_{time_of_day}": AMSR2_MPDI.values.ravel(),
        f"AMSR2_MPDI_{mpdi_band}_morning": pd.Series(_mpdi_mor),
        f"AMSR2_MPDI_{mpdi_band}_evening": pd.Series(_mpdi_eve),
        # f"AMSR2_{numerator_Hpol}/{denominator_Vpol}_{time_of_day}": fraction.values.ravel(),
        #
        # f"AMSR2_C1v_{time_of_day}": AMSR_C1v.values.ravel(),
        # f"AMSR2_C1h_{time_of_day}": AMSR_C1h.values.ravel(),
        # f"AMSR2_C2v_{time_of_day}": AMSR_C2v.values.ravel(),
        # f"AMSR2_C2h_{time_of_day}": AMSR_C2h.values.ravel(),
        # f"AMSR2_Xv_{time_of_day}": AMSR_Xv.values.ravel(),
        # f"AMSR2_Xh_{time_of_day}": AMSR_Xh.values.ravel(),
        # f"AMSR2_KUv_{time_of_day}": AMSR_KUv.values.ravel(),
        # f"AMSR2_KUh_{time_of_day}": AMSR_KUh.values.ravel(),
        # f"AMSR2_Kv_{time_of_day}": AMSR_Kv.values.ravel(),
        # f"AMSR2_Kh_{time_of_day}": AMSR_Kh.values.ravel(),
        # f"AMSR2_KAv_{time_of_day}": AMSR_KAv.values.ravel(),
        # f"AMSR2_KAh_{time_of_day}": AMSR_KAh.values.ravel(),
        # f"AMSR2_Wv_{time_of_day}": AMSR_Wv.values.ravel(),
        # f"AMSR2_Wh_{time_of_day}": AMSR_Wh.values.ravel(),
    })

    return data_df


# def main_processor(MODIS_LST,
#                    MODIS_NDVI,
#                    AMSR2,
#                    time_of_day: Literal["morning", "evening"],
#                    soil_range=[0, 0.2],
#                    veg_range=[0.5, 1],
#                    mpdi_band="x"
#                    ):
#     """
#     Main function that coregisters LST, NDVI and AMSR2 in space and time.
#     Calculates soil/veg. temperatures, and coarsens MODIS observations.
#     Please be understanding, there could be errors in the methods.
#     :param MODIS_LST: MODIS LST (MYD11) preferably cropped to whole AMSR2 extent.
#     :param MODIS_NDVI: MODIS_LST: MODIS NDVI (from MYD09 SR) preferably cropped to whole AMSR2 extent. Only daytime!
#     :param AMSR2: AMSR2 Level-1 BTs. Globall, day and nighttime observations
#     :param time_of_day: "morning" or "evening" in UTC!!
#     :param soil_range: NDVI range to classify soil
#     :param veg_range: NDVI range to classify vegetation
#     :return: pd.Dataframe that consists of all data
#     """
#
#     _MODIS_LST = morning_evening_passes(MODIS_LST, time_of_day=time_of_day)
#     _MODIS_NDVI = morning_evening_passes(MODIS_NDVI, time_of_day=time_of_day)
#     _AMSR2_data = morning_evening_passes(AMSR2, time_of_day=time_of_day)
#
#     if time_of_day == "evening":
#         MODIS_LST_match, MODIS_NDVI_match = match_MYD09_to_MYD11(_MODIS_LST, _MODIS_NDVI)
#     elif time_of_day == "morning":
#         MODIS_LST_match, MODIS_NDVI_match = _MODIS_LST,_MODIS_NDVI
#
#     common_AMSR2_data, common_MODIS_LST = common_observations(_AMSR2_data, MODIS_LST_match)
#     coarse_MODIS_LST = coarsen_highres(highres_da=common_MODIS_LST, lowres_da=common_AMSR2_data)
#     # TODO: this is a mess lets be honest, i will need to refactor this with logic
#     if time_of_day == "evening":
#         _, common_MODIS_NDVI = common_observations(_AMSR2_data, MODIS_NDVI_match)
#
#         soil_temp, veg_temp = threshold_ndvi(lst=common_MODIS_LST, ndvi=common_MODIS_NDVI,
#                                              soil_range=soil_range, veg_range=veg_range)
#         coarse_veg_temp = coarsen_highres(highres_da=veg_temp, lowres_da=common_AMSR2_data)
#         coarse_soil_temp = coarsen_highres(highres_da=soil_temp, lowres_da=common_AMSR2_data)
#
#     elif time_of_day == "morning":
#         coarse_soil_temp, coarse_veg_temp = (xr.full_like(common_AMSR2_data["scantime"],fill_value=1),
#                                              xr.full_like(common_AMSR2_data["scantime"],fill_value=1))
#
#     AMSR2_MPDI = mpdi(common_AMSR2_data, band=mpdi_band)
#     AMSR_LST = calc_Holmes_temp(common_AMSR2_data)
#
#     AMSR_C1v = common_AMSR2_data["bt_6.9V"]
#     AMSR_C1h = common_AMSR2_data["bt_6.9H"]
#     AMSR_C2v = common_AMSR2_data["bt_7.3V"]
#     AMSR_C2h = common_AMSR2_data["bt_7.3H"]
#     AMSR_Xv = common_AMSR2_data["bt_10.7V"]
#     AMSR_Xh = common_AMSR2_data["bt_10.7H"]
#     AMSR_KUv = common_AMSR2_data["bt_18.7V"]
#     AMSR_KUh = common_AMSR2_data["bt_18.7H"]
#     AMSR_Kv = common_AMSR2_data["bt_23.8V"]
#     AMSR_Kh = common_AMSR2_data["bt_23.8H"]
#     AMSR_KAv = common_AMSR2_data["bt_36.5V"]
#     AMSR_KAh = common_AMSR2_data["bt_36.5H"]
#     AMSR_Wv = common_AMSR2_data["bt_89.0V"]
#     AMSR_Wh = common_AMSR2_data["bt_89.0H"]
#     AMSR2_KUKA = MW_fraction(common_AMSR2_data)
#
#     data_df = pd.DataFrame({
#         f"MODIS_LST_{time_of_day}": coarse_MODIS_LST.values.ravel(),
#         f"MODIS_veg_temp_{time_of_day}": coarse_veg_temp.values.ravel(),
#         f"MODIS_soil_temp_{time_of_day}": coarse_soil_temp.values.ravel(),
#         f"AMSR2_LST_{time_of_day}": AMSR_LST.values.ravel(),
#         f"AMSR2_KUKA_{time_of_day}": AMSR2_KUKA.values.ravel(),
#         f"AMSR2_MPDI_{time_of_day}": AMSR2_MPDI.values.ravel(),
#
#         f"AMSR2_C1v_{time_of_day}": AMSR_C1v.values.ravel(),
#         f"AMSR2_C1h_{time_of_day}": AMSR_C1h.values.ravel(),
#         f"AMSR2_C2v_{time_of_day}": AMSR_C2v.values.ravel(),
#         f"AMSR2_C2h_{time_of_day}": AMSR_C2h.values.ravel(),
#         f"AMSR2_Xv_{time_of_day}": AMSR_Xv.values.ravel(),
#         f"AMSR2_Xh_{time_of_day}": AMSR_Xh.values.ravel(),
#         f"AMSR2_KUv_{time_of_day}": AMSR_KUv.values.ravel(),
#         f"AMSR2_KUh_{time_of_day}": AMSR_KUh.values.ravel(),
#         f"AMSR2_Kv_{time_of_day}": AMSR_Kv.values.ravel(),
#         f"AMSR2_Kh_{time_of_day}": AMSR_Kh.values.ravel(),
#         f"AMSR2_KAv_{time_of_day}": AMSR_KAv.values.ravel(),
#         f"AMSR2_KAh_{time_of_day}": AMSR_KAh.values.ravel(),
#         f"AMSR2_Wv_{time_of_day}": AMSR_Wv.values.ravel(),
#         f"AMSR2_Wh_{time_of_day}": AMSR_Wh.values.ravel(),
#     })
#
#     return data_df

##

landcover = "desert"
mpdi_band = "c2"
time_of_day = "morning"
numerator_Hpol = "c2"
denominator_Vpol = "ka"

x_axis_scatter  = f"AMSR2_MPDI_{mpdi_band}_"
y_axis_scatter  = f"AMSR2_MPDI_{mpdi_band}_"
# y_axis_scatter  = f"AMSR2_{numerator_Hpol}/{denominator_Vpol}_"

dates = pd.date_range(start="2018-01-01", end="2019-01-01", freq="MS")

if __name__=="__main__":

    monthly_boxplots(dates, landcover, time_of_day, x_axis_scatter, y_axis_scatter, ylim=[0,0.1])
    # monthly_scatter(dates, landcover, time_of_day, x_axis_scatter, y_axis_scatter,
    #                 numerator_Hpol = numerator_Hpol, denominator_Vpol=denominator_Vpol,
    #                 xlim=[0,0.06],ylim=[0.88,1.05])

