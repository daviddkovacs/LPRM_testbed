import pandas as pd
from LST.plot_functions import plot_hexbin, boxplot_timeseries, plot_modis_comparison
from LST.datacube_class import DATA_READER
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from datacube_utilities import (morning_evening_passes,coarsen_highres, common_observations)


if __name__=="__main__":

    time_start = "2018-01-01"
    time_stop = "2018-04-01"

    bbox = [
    -105.51503140246336,
    36.56150718001447,
    -104.5885313254013,
    37.121172994492596
  ]

    Data = DATA_READER(
        region="midwest",
        sensor="MODIS",
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
    MODIS_NDVI_cropped, MODIS_LST_cropped = MODIS_NDVI_cropped["NDVI"], MODIS_LST_cropped["LST"]

    plotdate = "2018-02-04T08:43:13"
    plot_modis_comparison(MODIS_NDVI_cropped, MODIS_LST_cropped, ndvi_time=plotdate,
                          lst_time=plotdate)
    plt.figure()
    AMSR2_LST.sel(time=plotdate, method="nearest").compute().plot.pcolormesh(x="lon", y="lat")
    plt.show(block=True)

##

    m_MODIS_LST, e_MODIS_LST = morning_evening_passes(MODIS_LST_cropped)
    m_AMSR2_LST, e_AMSR2_LST = morning_evening_passes(AMSR2_LST)

    common_m_AMSR2_LST, common_m_MODIS_LST = common_observations(m_AMSR2_LST, m_MODIS_LST,)
    common_e_AMSR2_LST, common_e_MODIS_LST = common_observations(e_AMSR2_LST, e_MODIS_LST,)

    coarse_m_MODIS_LST = coarsen_highres(highres_da=common_m_MODIS_LST,
                    lowres_da=common_m_AMSR2_LST)
    coarse_e_MODIS_LST = coarsen_highres(highres_da=common_e_MODIS_LST,
                    lowres_da=common_e_AMSR2_LST)


    plot_modis_comparison(MODIS_NDVI_cropped, common_e_MODIS_LST, ndvi_time=plotdate,
                          lst_time=plotdate)

    plt.figure()
    coarse_e_MODIS_LST.sel(time = plotdate,method="nearest").plot(x = "lon",y = "lat")
    plt.show()

    _coarse_e_MODIS_LST  = coarse_e_MODIS_LST.values.ravel()
    _coarse_m_MODIS_LST = coarse_m_MODIS_LST.values.ravel()
    _common_e_AMSR2_LST = common_e_AMSR2_LST.values.ravel()
    _common_m_AMSR2_LST = common_m_AMSR2_LST.values.ravel()

    data_df_m = pd.DataFrame({
                            "MODIS_LST_mor": coarse_m_MODIS_LST.values.ravel(),
                            "AMSR2_LST_mor": common_m_AMSR2_LST.values.ravel()},
                           )
    data_df_e = pd.DataFrame({"MODIS_LST_eve": coarse_e_MODIS_LST.values.ravel(),
                            "AMSR2_LST_eve": common_e_AMSR2_LST.values.ravel(),
                              },)
    plot_hexbin(data_df_e,"MODIS_LST_eve", "AMSR2_LST_eve")
    plot_hexbin(data_df_m,"MODIS_LST_mor", "AMSR2_LST_mor")
