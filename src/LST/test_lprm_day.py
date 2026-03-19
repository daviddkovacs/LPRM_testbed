import matplotlib.pyplot as plt

from mpdi_differences import load_AMSR2_daily, retrieve_LPRM, calc_Holmes_temp


##
if __name__=="__main__":

    bbox = [-180, -90, 180, 90]
    time_start = "2018-01-01"
    time_stop = "2019-01-01"
    bandlist = ["c2", "x", "ku"]

    AMSR2_DAY, AMSR2_NIGHT = load_AMSR2_daily(bbox = bbox,time_start=time_start,time_stop=time_stop)
    HOLMES_T_NIGHT, HOLMES_T_DAY = calc_Holmes_temp(AMSR2_NIGHT), calc_Holmes_temp(AMSR2_DAY)
    NEW_T_DAY = AMSR2_DAY["bt_36.5V"] * 0.38 + 185

##
    band_current = "x"
    SM_NIGHT, VOD_NIGHT,_ = retrieve_LPRM(TB_DATASET=AMSR2_NIGHT, SURFACE_T=HOLMES_T_NIGHT, band=band_current)

    SM_DAY, VOD_DAY, _ = retrieve_LPRM(TB_DATASET=AMSR2_DAY, SURFACE_T=NEW_T_DAY, band=band_current)
    SM_DAY_sahel, VOD_DAY_sahel, _ = retrieve_LPRM(TB_DATASET=AMSR2_DAY, SURFACE_T=NEW_T_DAY, band=band_current)
    SM_DAY_ref, VOD_DAY_ref, _ = retrieve_LPRM(TB_DATASET=AMSR2_DAY, SURFACE_T=HOLMES_T_DAY, band=band_current)


##
    lat , lon = 39.505976, -89.551469
    plt.figure(figsize=(20,4))
    SM_DAY.sel(lat = lat, lon= lon, method="nearest").plot(label ="Day test")
    SM_DAY_ref.sel(lat =lat, lon=lon, method="nearest").plot(label ="Day Holmes")
    SM_NIGHT.sel(lat = lat, lon= lon, method="nearest").plot(label ="Night Holmes")
    plt.legend()
    plt.show()

##
    compression_settings = {"zlib": True, "complevel": 5}

    encoding_dict = {"sm": compression_settings}

    SM_DAY.to_netcdf("/home/ddkovacs/personal_data/lprm_daytime/"
                     f"SM{band_current}_day.nc", encoding={"sm": compression_settings})
    VOD_DAY.to_netcdf("/home/ddkovacs/personal_data/lprm_daytime/"
                     f"VOD{band_current}_DAY", encoding={"vod": compression_settings})


##
    # SM_DAY_ref.to_netcdf("/home/ddkovacs/personal_data/lprm_daytime/"
    #                  f"SM{band_current}_DAY_ref.nc", encoding={"sm": compression_settings})
    # VOD_DAY_ref.to_netcdf("/home/ddkovacs/personal_data/lprm_daytime/"
    #                  f"VOD{band_current}_DAY_ref.nc", encoding={"vod": compression_settings})
