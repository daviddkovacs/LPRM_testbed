import pandas as pd
from LST.plot_functions import plot_hexbin, plot_modis_comparison
from LST.datacube_class import DATA_READER
from datacube_utilities import (morning_evening_passes,coarsen_highres, common_observations, mpdi, calc_Holmes_temp, KuKa, threshold_ndvi)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


if __name__=="__main__":

    time_start = "2018-03-01"
    time_stop = "2018-09-01"

    bbox =[
    -104.28143790401508,
    36.05666157843547,
    -103.82161297993213,
    36.40500798732522
  ]

    Data = DATA_READER(
        region="midwest",
        bbox= bbox,
        time_start=time_start,
        time_stop=time_stop,
    )

##
    soil_range = [0, 0.2]
    veg_range = [0.5, 1]

    mpdi_band = "x"
    AMSR2_data = Data.AMSR2_BT

    MODIS_NDVI_cropped, MODIS_LST_cropped = Data.match_AMSR2_extent()

    plotdate = "2018-01-01T08:43:13"
    # plot_modis_comparison(MODIS_NDVI_cropped, MODIS_LST_cropped, ndvi_time=plotdate,
    #                       lst_time=plotdate)
    # plt.figure()
    # AMSR2_data.sel(time=plotdate, method="nearest").compute().plot.pcolormesh(x="lon", y="lat")
    # plt.show()

##  # TODO: will need to properly refactor, and revise this section.
    time_of_day = "evening" # For NDVI only the evening overpass works over the mdiwest, as it is 3pm there.
    _MODIS_LST = morning_evening_passes(MODIS_LST_cropped, time_of_day=time_of_day)
    _MODIS_NDVI = morning_evening_passes(MODIS_NDVI_cropped, time_of_day=time_of_day)
    _AMSR2_data = morning_evening_passes(AMSR2_data, time_of_day=time_of_day)

    common_AMSR2_data, common_MODIS_LST = common_observations(_AMSR2_data, _MODIS_LST)
    _, common_MODIS_NDVI = common_observations(_AMSR2_data, _MODIS_NDVI)
    COMMON_MODIS_LST, COMMON_MODIS_NDVI = common_observations(_MODIS_LST, _MODIS_NDVI, method="exact")

    def calc_soil_veg_t(LST, NDVI ,soil_range = [0, 0.2],veg_range = [0.5, 1] ):

        # _common_MODIS_NDVI,_common_MODIS_LST  = common_observations(NDVI, LST) # The order is important as NDVI has less obs.
        soil_temp, veg_temp = threshold_ndvi(lst=LST,ndvi=NDVI, soil_range=soil_range, veg_range= veg_range)
        return soil_temp, veg_temp


    soil_temp, veg_temp = calc_soil_veg_t(COMMON_MODIS_LST,COMMON_MODIS_NDVI)


    coarse_veg_temp = coarsen_highres(highres_da=veg_temp,lowres_da=common_AMSR2_data)
    coarse_soil_temp = coarsen_highres(highres_da=soil_temp,lowres_da=common_AMSR2_data)
    coarse_MODIS_LST = coarsen_highres(highres_da=common_MODIS_LST,lowres_da=common_AMSR2_data)
    coarse_MODIS_NDVI = coarsen_highres(highres_da=common_MODIS_NDVI,lowres_da=common_AMSR2_data)


    _AMSR2_data, coarse_soil_temp = common_observations(_AMSR2_data, coarse_soil_temp, method="nearest")
    _AMSR2_data, coarse_veg_temp = common_observations(_AMSR2_data, coarse_veg_temp, method="nearest")

    AMSR2_MPDI = mpdi(common_AMSR2_data, band=mpdi_band)
    AMSR_LST = calc_Holmes_temp(_AMSR2_data)
    AMSR2_KUKA = KuKa(common_AMSR2_data)

    data_df = pd.DataFrame({
                            # f"MODIS_LST_{time_of_day}": coarse_MODIS_LST.values.ravel(),
                            # f"MODIS_NDVI_{time_of_day}": coarse_MODIS_NDVI.values.ravel(),
                            f"soil_temp_{time_of_day}": coarse_soil_temp.values.ravel(),
                            f"veg_temp_{time_of_day}": coarse_veg_temp.values.ravel(),
                            f"AMSR2_LST_{time_of_day}": AMSR_LST.values.ravel(),}
                            # f"AMSR2_KUKA_{time_of_day}": AMSR2_KUKA.values.ravel(),
                            # f"AMSR2_MPDI_{time_of_day}": AMSR2_MPDI.values.ravel()},
                           )


    # plot_hexbin(data_df,f"MODIS_LST_{time_of_day}", f"AMSR2_LST_{time_of_day}", utc_timeofday=time_of_day)
    plot_hexbin(data_df,f"soil_temp_{time_of_day}", f"AMSR2_LST_{time_of_day}", utc_timeofday=time_of_day)
    plot_hexbin(data_df,f"veg_temp_{time_of_day}", f"AMSR2_LST_{time_of_day}", utc_timeofday=time_of_day)
