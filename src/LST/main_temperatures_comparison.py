import pandas as pd

from LST.datacube_utilities import match_MYD09_to_MYD11
from LST.plot_functions import plot_hexbin
from LST.datacube_class import DATA_READER
from datacube_utilities import (morning_evening_passes,coarsen_highres, common_observations, mpdi, calc_Holmes_temp, KuKa, threshold_ndvi)
import matplotlib

matplotlib.use('TkAgg')

if __name__=="__main__":

    time_start = "2018-02-01"
    time_stop = "2018-08-10"

    bbox =  [
    -105.03358126807578,
    36.42675231936566,
    -103.90454819929712,
    37.28711042016215
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

##
    time_of_day = "morning" # For NDVI only the evening overpass works over the mdiwest, as it is 3pm there.

    _MODIS_LST = morning_evening_passes(MODIS_LST_cropped, time_of_day=time_of_day)
    _MODIS_NDVI = morning_evening_passes(MODIS_NDVI_cropped, time_of_day=time_of_day)
    _AMSR2_data = morning_evening_passes(AMSR2_data, time_of_day=time_of_day)

    MODIS_LST_match, MODIS_NDVI_match = match_MYD09_to_MYD11(_MODIS_LST, _MODIS_NDVI)

    common_AMSR2_data, common_MODIS_LST = common_observations(_AMSR2_data, MODIS_LST_match)
    _, common_MODIS_NDVI = common_observations(_AMSR2_data, MODIS_NDVI_match)

    soil_temp, veg_temp =  threshold_ndvi(lst= common_MODIS_LST,ndvi = common_MODIS_NDVI,
                                          soil_range=[0,0.2], veg_range=[0.6,1])

    coarse_MODIS_LST = coarsen_highres(highres_da=common_MODIS_LST,lowres_da=common_AMSR2_data)
    coarse_veg_temp = coarsen_highres(highres_da=veg_temp,lowres_da=common_AMSR2_data)
    coarse_soil_temp = coarsen_highres(highres_da=soil_temp,lowres_da=common_AMSR2_data)

    AMSR2_MPDI = mpdi(common_AMSR2_data, band=mpdi_band)
    AMSR_LST = calc_Holmes_temp(_AMSR2_data)
    AMSR2_KUKA = KuKa(common_AMSR2_data)

    data_df = pd.DataFrame({
                            f"MODIS_LST_{time_of_day}": coarse_MODIS_LST.values.ravel(),
                            f"MODIS_veg_temp_{time_of_day}": coarse_veg_temp.values.ravel(),
                            f"MODIS_soil_temp_{time_of_day}": coarse_soil_temp.values.ravel(),
                            f"AMSR2_LST_{time_of_day}": AMSR_LST.values.ravel(),}
                            # f"AMSR2_KUKA_{time_of_day}": AMSR2_KUKA.values.ravel(),
                            # f"AMSR2_MPDI_{time_of_day}": AMSR2_MPDI.values.ravel()},
                           )


    plot_hexbin(data_df,f"MODIS_LST_{time_of_day}", f"AMSR2_LST_{time_of_day}", utc_timeofday=time_of_day)
    plot_hexbin(data_df,f"MODIS_veg_temp_{time_of_day}", f"AMSR2_LST_{time_of_day}", utc_timeofday=time_of_day)
    plot_hexbin(data_df,f"MODIS_soil_temp_{time_of_day}", f"AMSR2_LST_{time_of_day}", utc_timeofday=time_of_day)
    x = 1



    # plot_hexbin(data_df,f"MODIS_LST_{time_of_day}", f"AMSR2_KUKA_{time_of_day}",
    #             ylim = [None,None],
    #             xlim = [None,None],
    #             utc_timeofday=time_of_day)
    # plot_hexbin(data_df,f"MODIS_LST_{time_of_day}", f"AMSR2_MPDI_{time_of_day}",
    #             ylim = [None,None],
    #             xlim = [None,None],
    #             utc_timeofday=time_of_day)
