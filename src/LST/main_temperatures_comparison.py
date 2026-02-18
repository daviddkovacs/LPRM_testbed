import pandas as pd
from LST.plot_functions import plot_hexbin, boxplot_timeseries, plot_modis_comparison
from LST.datacube_class import DATA_READER
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

if __name__=="__main__":

    time_start = "2018-01-01"
    time_stop = "2018-01-10"

    bbox = [
    -104.89859042693736,
    36.22330318012534,
    -103.49362827363801,
    37.49045690783343
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

    NDVI_cropped, LST_cropped = Data.match_AMSR2_extent()

    plotdate = "2018-01-08T08:43:13"
    plot_modis_comparison(NDVI_cropped["NDVI"], LST_cropped["LST"], ndvi_time=plotdate,
                          lst_time=plotdate)
    plt.figure()
    Data.AMSR2_LST.sel(time=plotdate, method="nearest").compute().plot.pcolormesh(x="lon", y="lat")
    plt.show(block=True)

##
    date = "2024-08-01"

    Data.temperatures_dashboard(bbox=bbox,date=date, scatter_x= "veg_temp", )
    Data.plot_AMSR2(bbox=bbox,date=date)


##

    fig = boxplot_timeseries(complete_df, mpdi_band=mpdi_band)
    plt.show()

    # with open("/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/"
    #                 "LPRM/07_debug/daytime_retrieval/LST/figs/fig1.pkl", "wb") as f:
    #
    #     pickle.dump(fig,f)