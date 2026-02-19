import pandas as pd
from LST.plot_functions import plot_hexbin, plot_modis_comparison
from LST.datacube_class import DATA_READER
from datacube_utilities import (morning_evening_passes,coarsen_highres, common_observations)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


if __name__=="__main__":

    time_start = "2018-01-01"
    time_stop = "2018-01-05"

    bbox = [
    -105.51503140246336,
    36.56150718001447,
    -104.5885313254013,
    37.121172994492596
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
    AMSR2_LST = Data.AMSR2_LST

    MODIS_NDVI_cropped, MODIS_LST_cropped = Data.match_AMSR2_extent()

    plotdate = "2018-01-01T08:43:13"
    plot_modis_comparison(MODIS_NDVI_cropped, MODIS_LST_cropped, ndvi_time=plotdate,
                          lst_time=plotdate)
    plt.figure()
    AMSR2_LST.sel(time=plotdate, method="nearest").compute().plot.pcolormesh(x="lon", y="lat")
    plt.show(block=True)

##
    time_of_day = "morning"
    _MODIS_LST = morning_evening_passes(MODIS_LST_cropped, time_of_day=time_of_day)
    _AMSR2_LST = morning_evening_passes(AMSR2_LST, time_of_day=time_of_day)

    common_AMSR2_LST, common_MODIS_LST = common_observations(_AMSR2_LST, _MODIS_LST)

    coarse_MODIS_LST = coarsen_highres(highres_da=common_MODIS_LST,
                    lowres_da=common_AMSR2_LST)


    plot_modis_comparison(MODIS_NDVI_cropped, common_MODIS_LST, ndvi_time=plotdate,
                          lst_time=plotdate)

    plt.figure()
    coarse_MODIS_LST.sel(time = plotdate,method="nearest").plot(x = "lon",y = "lat")
    plt.show(block = True)


    data_df_m = pd.DataFrame({
                            f"MODIS_LST_{time_of_day}": coarse_MODIS_LST.values.ravel(),
                            f"AMSR2_LST_{time_of_day}": common_AMSR2_LST.values.ravel()},
                           )
    plot_hexbin(data_df_m,f"MODIS_LST_{time_of_day}", f"AMSR2_LST_{time_of_day}")
