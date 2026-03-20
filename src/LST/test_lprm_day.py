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
    NEW_T_DAY = AMSR2_DAY["bt_36.5V"] * 0.84 + 62.33

##
    band_current = "x"
    SM_NIGHT, VOD_NIGHT,_ = retrieve_LPRM(TB_DATASET=AMSR2_NIGHT, SURFACE_T=HOLMES_T_NIGHT, band=band_current)

    _, _,T_sim_day = retrieve_LPRM(TB_DATASET=AMSR2_DAY, SURFACE_T=HOLMES_T_DAY,
                                          SM_input=SM_NIGHT, VOD_input=VOD_NIGHT,band=band_current)

    SM_DAY, VOD_DAY, _ = retrieve_LPRM(TB_DATASET=AMSR2_DAY, SURFACE_T=HOLMES_T_DAY, band=band_current)
    SM_DAY_new, VOD_DAY_new, _ = retrieve_LPRM(TB_DATASET=AMSR2_DAY, SURFACE_T=T_sim_day, band=band_current)


##
    lat , lon =-35.350097, 146.656135

    plt.figure(figsize=(20,4))
    SM_DAY.sel(lat = lat, lon = lon, method = "nearest").plot(label ="SM_DAY")
    SM_DAY_new.sel(lat = lat, lon = lon, method = "nearest").plot(label ="SM_DAY_new")
    SM_NIGHT.sel(lat = lat, lon = lon, method = "nearest").plot(label ="SM_NIGHT")
    # (SM_DAY_new - SM_NIGHT).sel(lat = lat, lon = lon, method = "nearest").plot(label ="SM_DAY_new - SM_NIGHT")
    plt.legend()
    plt.show()

##



