import os.path
import matplotlib.pyplot as plt
from LST.mpdi_differences import load_TB_daily, retrieve_LPRM, calc_Holmes_temp, date_pattern_lut, file_pattern_lut
import xarray as xr
import numpy as np


##
if __name__=="__main__":

    bbox = [-180, -90, 180, 90]
    time_start = "2024-01-01"
    time_stop = "2024-12-01"
    bandlist = ["c1","c2", "x", "ku"]
    sensor = "AMSR2"

    AMSR2_DAY, AMSR2_NIGHT = load_TB_daily(bbox=bbox, time_start=time_start, time_stop=time_stop,
                                           date_pattern = date_pattern_lut[sensor],
                                           file_pattern=file_pattern_lut[sensor])

    HOLMES_T_NIGHT, HOLMES_T_DAY = calc_Holmes_temp(AMSR2_NIGHT), calc_Holmes_temp(AMSR2_DAY)

    ##
    band_current = "c1"
    minimum_mpdi = 0.010

    path_aux_t = (f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/lprm_testing"
                  f"/T_aux/{band_current}_daytime_LST_regression.nc")
    daytime_stats = xr.open_dataset(path_aux_t)

    T_KA = AMSR2_DAY["bt_36.5V"]

    slope = daytime_stats["slope"]
    intercept = daytime_stats["intercept"]

    T_DAYTIME = (T_KA * slope + intercept).compute()

##
    SM_NIGHT_ref, VOD_NIGHT_ref ,_ = retrieve_LPRM(TB_DATASET=AMSR2_NIGHT,
                                                   SURFACE_T=HOLMES_T_NIGHT,
                                                   band=band_current)


    SM_DAY_ref, VOD_DAY_ref, _ = retrieve_LPRM(TB_DATASET=AMSR2_DAY,
                                       SURFACE_T=HOLMES_T_DAY,
                                       band=band_current)

    SM_DAY_regression, VOD_DAY_regression, _ = retrieve_LPRM(TB_DATASET=AMSR2_DAY,
                                                             SURFACE_T=T_DAYTIME,
                                                             band=band_current)


    ##
    path_shares = (f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/"
                   f"lprm_testing/SM/MPDI_{minimum_mpdi}")

    path_out = path_shares
    compression_settings = {"zlib": True, "complevel": 5}

    # SM_NIGHT_ref.to_netcdf(os.path.join(path_out, f"SM{band_current}_NIGHT_ref.nc"), encoding={"sm": compression_settings})
    # VOD_NIGHT_ref.to_netcdf(os.path.join(path_out, f"VOD{band_current}_NIGHT_ref.nc"), encoding={"vod": compression_settings})
    #
    # SM_DAY_ref.to_netcdf(os.path.join(path_out, f"SM{band_current}_DAY_ref.nc"),encoding={"sm": compression_settings})
    # VOD_DAY_ref.to_netcdf(os.path.join(path_out, f"VOD{band_current}_DAY_ref.nc"), encoding={"vod": compression_settings})
    #
    # SM_DAY_regression.to_netcdf(os.path.join(path_out, f"SM{band_current}_DAY_regression.nc"), encoding={"sm": compression_settings})
    # VOD_DAY_regression.to_netcdf(os.path.join(path_out, f"VOD{band_current}_DAY_regression.nc"), encoding={"vod": compression_settings})



##
    t = 15
    sm_ref_1 = SM_DAY_ref.isel(time = t)
    sm_regression_1 = SM_DAY_regression.isel(time = t)
    temp = T_DAYTIME.isel(time = t)
    holmes = HOLMES_T_DAY.isel(time = t)

    bias_map = SM_DAY_ref.mean(dim= "time") - SM_DAY_regression.mean(dim= "time")
    _bias = bias_map.where((bias_map>0.1))

    bias = _bias.where(_bias>0)
    temp = temp.where(_bias>0)
    holmes = holmes.where(_bias>0)
    _slope = slope.where(_bias>0)
    _intercept = intercept.where(_bias>0)

    bias_values = bias.values.ravel()
    temp_values = temp.values.ravel()
    holmes_values = holmes.values.ravel()
    _slope_values = _slope.values.ravel()
    _intercept_values = _intercept.values.ravel()

    # 2. Create a single boolean mask where NO array has a NaN
    valid_mask = (
            ~np.isnan(bias_values) &
            ~np.isnan(temp_values) &
            ~np.isnan(holmes_values) &
            ~np.isnan(_slope_values) &
            ~np.isnan(_intercept_values)
    )

    # 3. Apply the exact same mask to every array
    bias_filter = bias_values[valid_mask]
    temp_filter = temp_values[valid_mask]
    holmes_filter = holmes_values[valid_mask]
    _slope_filter = _slope_values[valid_mask]
    _intercept_filter = _intercept_values[valid_mask]
##
    import pandas as pd

    df = pd.DataFrame({
        "bias":bias_filter,
        "t":temp_filter,
        "holmes":holmes_filter,
        "t-holmes":temp_filter-holmes_filter,
        "slope":_slope_filter,
        "intercept":_intercept_filter,
                       })

##
    plt.figure()
    # plt.plot(_intercept_filter)
    # plt.plot(temp_filter)
    bias_map.plot()
    plt.show()



