import pandas as pd
from LST.plot_functions import plot_hexbin, boxplot_timeseries
from LST.datacube_class import DATA_READER
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
if __name__=="__main__":

    time_start = "2023-01-01"
    time_stop = "2026-01-01"

    bbox =  [
    -2.94988870276606,
    14.13623145787058,
    -2.237090476608074,
    14.676788211060355
  ]

    Data = DATA_READER(
        region="sahel",
        bbox= bbox,
        time_start=time_start,
        time_stop=time_stop,
    )

##

    soil_range = [0, 0.2]
    veg_range = [0.5, 1]
    mpdi_band = "x"
    dflist = []
    timesteps = Data.DATACUBES_L1["SLSTR"].time

    for d in timesteps:
        # try:
        print(f"processing: {d.values}")
        dflist.append(Data.process_date(date = d,  bbox= bbox,
                                        soil_range=soil_range,
                                        veg_range=veg_range,
                                        mpdi_band="x"))
        # except Exception as e:
        #     print(e)

    complete_df = pd.concat(dflist)

    # plt.close("all")
    plot_hexbin(complete_df,"mpdi", "veg_temp",xlim= [0,0.03], ylim = [273,320])
    plot_hexbin(complete_df,"veg_temp", "tsurf_ka", xlim= [0,0.1], ylim = [273,320])
    plot_hexbin(complete_df,"kuka","soil_temp", xlim= [0.9,1], ylim = [273,320])

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