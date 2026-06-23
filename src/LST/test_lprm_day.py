import os.path
import matplotlib.pyplot as plt
from LST.mpdi_differences import load_TB_daily, retrieve_LPRM, calc_Holmes_temp, date_pattern_lut, file_pattern_lut
import xarray as xr
import numpy as np
import pandas as pd
from LST.datacube_utilities import crop2roi

nested_group = {"SMOS" : "42.5",
                "SMAP" :None,
                "AMSR2": None}

if __name__=="__main__":

    bbox = [-180, -90, 180, 90]
    time_start = "2024-01-01"
    time_stop = "2024-12-01"
    bandlist = ["l","c1","c2", "x", "ku"]
    sensor = "SMOS"

    TB_DAY, TB_NIGHT = load_TB_daily(bbox=bbox, time_start=time_start, time_stop=time_stop,
                                     sensor=sensor,file_pattern=file_pattern_lut[sensor],
                                     date_pattern=date_pattern_lut[sensor],nested_group_name=nested_group[sensor]
                                     )


    if sensor.upper() not in ["SMOS", "SMAP"]:
        HOLMES_T_NIGHT, HOLMES_T_DAY = calc_Holmes_temp(TB_NIGHT, sensor=sensor), calc_Holmes_temp(TB_DAY, sensor=sensor)
    elif sensor.upper() in ["SMAP","SMOS"]:
        HOLMES_T_DAY, HOLMES_T_NIGHT = load_TB_daily(bbox=bbox, time_start=time_start, time_stop=time_stop,
                                         sensor=sensor, file_pattern=f"{sensor.lower()}_combined_*.nc",
                                             path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/L_Band_Temps/temps/10k",
                                             date_pattern = date_pattern_lut[sensor], nested_group_name="Ka"
                                         )
    ##
    band_current = "l"
    minimum_mpdi = 0.010

    path_aux_t = (f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/lprm_testing"
                  f"/T_aux/{band_current.upper()}_daytime_LST_regression.nc")
    daytime_stats = xr.open_dataset(path_aux_t)

    if sensor.upper() not in ["SMAP","SMOS"]:
        T_KA = TB_DAY["bt_36.5V"]
    elif sensor.upper() in ["SMAP","SMOS"]:
        T_KA = HOLMES_T_DAY["bt_vertical"]
##
    slope = daytime_stats["slope"]
    intercept = daytime_stats["intercept"]

    T_DAYTIME = (T_KA * slope + intercept).compute()

##
    SM_NIGHT_ref, VOD_NIGHT_ref ,_ = retrieve_LPRM(TB_DATASET=TB_NIGHT,
                                                   SURFACE_T=T_KA,
                                                   sensor=sensor,
                                                   band=band_current)


    SM_DAY_ref, VOD_DAY_ref, _ = retrieve_LPRM(TB_DATASET=TB_DAY,
                                       SURFACE_T=T_KA,
                                               sensor=sensor,
                                               band=band_current)

    SM_DAY_regression, VOD_DAY_regression, _ = retrieve_LPRM(TB_DATASET=TB_DAY,
                                                             SURFACE_T=T_DAYTIME,
                                                             band=band_current,
                                                             sensor=sensor,
                                                             )

    ##
    path_shares = (f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/"
                   f"lprm_testing/SM/MPDI_{minimum_mpdi}")

    # path_out = path_shares
    path_out = "/home/ddkovacs/Desktop"
    compression_settings = {"zlib": True, "complevel": 5}

    SM_NIGHT_ref.to_netcdf(os.path.join(path_out, f"SM{band_current.upper()}_NIGHT_ref.nc"), encoding={"sm": compression_settings})
    VOD_NIGHT_ref.to_netcdf(os.path.join(path_out, f"VOD{band_current.upper()}_NIGHT_ref.nc"), encoding={"vod": compression_settings})

    SM_DAY_ref.to_netcdf(os.path.join(path_out, f"SM{band_current.upper()}_DAY_ref.nc"),encoding={"sm": compression_settings})
    VOD_DAY_ref.to_netcdf(os.path.join(path_out, f"VOD{band_current.upper()}_DAY_ref.nc"), encoding={"vod": compression_settings})

    SM_DAY_regression.to_netcdf(os.path.join(path_out, f"SM{band_current.upper()}_DAY_regression.nc"), encoding={"sm": compression_settings})
    VOD_DAY_regression.to_netcdf(os.path.join(path_out, f"VOD{band_current.upper()}_DAY_regression.nc"), encoding={"vod": compression_settings})



##

  #   bbox_siberia = [
  #   96.88615696455196,
  #   54.25554741270571,
  #   114.06797376493057,
  #   62.907052962598044
  # ]

    bbox_siberia = [-180,-90,180,90]

    SM_DAY_ref_sib = crop2roi(SM_DAY_ref,bbox_siberia)
    SM_DAY_regression_sib = crop2roi(SM_DAY_regression,bbox_siberia)
    AMSR2_DAY_sib = crop2roi(AMSR2_DAY,bbox_siberia)
    slope_sib = crop2roi(slope,bbox_siberia)
    intercept_sib = crop2roi(intercept,bbox_siberia)
    T_DAYTIME_sib = crop2roi(T_DAYTIME,bbox_siberia)
    HOLMES_T_DAY_sib = crop2roi(HOLMES_T_DAY,bbox_siberia)

    MPDI_stack = ((AMSR2_DAY_sib["bt_6.9V"] - AMSR2_DAY_sib["bt_6.9H"])
                  / (AMSR2_DAY_sib["bt_6.9V"] + AMSR2_DAY_sib["bt_6.9H"]))

    bias_map = SM_DAY_ref_sib.mean(dim= "time") - SM_DAY_regression_sib.mean(dim= "time")
    abs_bias_stack = bias_map.broadcast_like(T_DAYTIME_sib)
    slope_stack = slope_sib.broadcast_like(T_DAYTIME_sib)
    intercept_stack = intercept_sib.broadcast_like(T_DAYTIME_sib)

    bias_values = abs_bias_stack.values.ravel()
    temp_values = T_DAYTIME_sib.values.ravel()
    holmes_values = HOLMES_T_DAY_sib.values.ravel()
    _slope_values = slope_stack.values.ravel()
    _intercept_values = intercept_stack.values.ravel()
    _MPDI_values = MPDI_stack.values.ravel()

    valid_mask = (
            ~np.isnan(bias_values) &
            ~np.isnan(temp_values) &
            ~np.isnan(holmes_values) &
            ~np.isnan(_MPDI_values)&
            ~np.isnan(_slope_values) &
            ~np.isnan(_intercept_values)
    )

    bias_filter = bias_values[valid_mask]
    temp_filter = temp_values[valid_mask]
    holmes_filter = holmes_values[valid_mask]
    _MPDI_filter = _MPDI_values[valid_mask]
    _slope_filter = _slope_values[valid_mask]
    _intercept_filter = _intercept_values[valid_mask]

    _df = pd.DataFrame({
        "bias":bias_filter,
        # "t":temp_filter,
        # "holmes":holmes_filter,
        "t-holmes":(temp_filter-holmes_filter),
        "slope":_slope_filter,
        "intercept":_intercept_filter,
        "MPDI":_MPDI_filter,
                       })

    # df = _df.where((_df["slope"]>1.00) & (
    # _df["intercept"]<0.001)).dropna()

##

    bias_filter = bias_map.where(
        (abs(te-holmes)<5) &
        (MPDI > 0.01)
    )
    plt.figure(figsize=(15,10))
    bias_filter.plot()
    plt.title("Bias filtered")
    plt.show()

    plt.figure(figsize=(15,10))
    abs_bias.plot(vmin=-0.25, vmax = 0.25,cmap = "coolwarm")
    plt.title("Bias ")
    plt.show()
##

    fig = plt.figure(figsize=(16, 6))

    df = _df.where((_df["slope"]>1.00) & (_df["intercept"]<0.001)).dropna()

    plt.scatter(
        df['slope'],
        df['bias'],
        c=df['intercept'],
        cmap='coolwarm',
        alpha=0.7,
        # vmin=-0.5,
        # vmax=0.3,
        s=15
    )
    plt.xlabel('slope')
    plt.ylabel('bias')
    plt.title('2D Heatmap View: Bias across Slope and Intercept')
    plt.colorbar(label='intercept')

    plt.tight_layout()
    plt.show()

