import pandas as pd
from LST.plot_functions import plot_hexbin, plot_modis_comparison
from LST.datacube_class import DATA_READER
from datacube_utilities import (morning_evening_passes,coarsen_highres, common_observations)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


if __name__=="__main__":

    time_start = "2018-01-01"
    time_stop = "2018-03-01"

    bbox =  [
    -104.7351137559069,
    35.95693111917318,
    -103.08335275645139,
    36.66648495873841
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
    plt.show()

##
    time_of_day = "evening"
    _MODIS_LST = morning_evening_passes(MODIS_LST_cropped, time_of_day=time_of_day)
    _AMSR2_LST = morning_evening_passes(AMSR2_LST, time_of_day=time_of_day)

    common_AMSR2_LST, common_MODIS_LST = common_observations(_AMSR2_LST, _MODIS_LST)

    coarse_MODIS_LST = coarsen_highres(highres_da=common_MODIS_LST,
                    lowres_da=common_AMSR2_LST)


    data_df_m = pd.DataFrame({
                            f"MODIS_LST_{time_of_day}": coarse_MODIS_LST.values.ravel(),
                            f"AMSR2_LST_{time_of_day}": common_AMSR2_LST.values.ravel()},
                           )
    plot_hexbin(data_df_m,f"MODIS_LST_{time_of_day}", f"AMSR2_LST_{time_of_day}")
