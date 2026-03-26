import matplotlib.pyplot as plt
from mpdi_differences import load_TB_daily, retrieve_LPRM, calc_Holmes_temp
import xarray as xr
from plot_functions import world_map

##
if __name__=="__main__":

    bbox = [-180, -90, 180, 90]
    time_start = "2018-01-01"
    time_stop = "2019-01-01"
    bandlist = ["c2", "x", "ku"]

    AMSR2_DAY, AMSR2_NIGHT = load_TB_daily(bbox=bbox, time_start=time_start, time_stop=time_stop,
                                           sensor="GMI", file_pattern="gmi_l1bt_*.nc")
    HOLMES_T_NIGHT, HOLMES_T_DAY = calc_Holmes_temp(AMSR2_NIGHT), calc_Holmes_temp(AMSR2_DAY)

    ##
    band_current = "x"

    path_aux_t = (f"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/lprm_testing"
                  f"/T_aux/Daytime_T_aux_{band_current}.nc")
    daytime_stats = xr.open_dataset(path_aux_t)

    T_KA = AMSR2_DAY["bt_36.5V"]

    slope = daytime_stats["slope"]
    intercept = daytime_stats["intercept"]

    T_DAYTIME = (T_KA * slope + intercept).compute()

##
    SM_NIGHT, VOD_NIGHT,_ = retrieve_LPRM(TB_DATASET=AMSR2_NIGHT,
                                          SURFACE_T=HOLMES_T_NIGHT,
                                          band=band_current)

    _, _,T_sim_day = retrieve_LPRM(TB_DATASET=AMSR2_DAY,
                                   SURFACE_T=HOLMES_T_DAY,
                                   SM_input=SM_NIGHT,
                                   VOD_input=VOD_NIGHT,
                                   band=band_current)

    SM_DAY, VOD_DAY, _ = retrieve_LPRM(TB_DATASET=AMSR2_DAY,
                                       SURFACE_T=HOLMES_T_DAY,
                                       band=band_current)

    SM_DAY_regression, VOD_DAY_regression, _ = retrieve_LPRM(TB_DATASET=AMSR2_DAY,
                                                             SURFACE_T=T_DAYTIME,
                                                             band=band_current)


##
    lat , lon = -32.752114, 146.7064

    plt.figure(figsize=(20,4))
    SM_DAY.sel(lat = lat, lon = lon, method = "nearest").plot(label ="SM_DAY")
    SM_DAY_regression.sel(lat = lat, lon = lon, method ="nearest").plot(label ="SM_DAY_regression")
    SM_NIGHT.sel(lat = lat, lon = lon, method = "nearest").plot(label ="SM_NIGHT")
    plt.show()

##
    compression_settings = {"zlib": True, "complevel": 5}

    encoding_dict = {"sm": compression_settings}

    SM_NIGHT.to_netcdf("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/07_debug/daytime_retrieval/MPDI_trick/lprm_testing/SM/"
                     f"SM{band_current}_NIGHT_ref.nc", encoding={"sm": compression_settings})




