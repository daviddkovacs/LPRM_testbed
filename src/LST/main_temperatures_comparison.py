import pandas as pd
from LST.plot_functions import plot_hexbin, boxplot_timeseries
from LST.SLSTR_AMSR2_reader import SLSTR_AMSR2_DC
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

if __name__=="__main__":

    time_start = "2024-01-01"
    time_stop = "2025-01-01"

    Data = SLSTR_AMSR2_DC(
        region="ceu",
        time_start=time_start,
        time_stop=time_stop,
    )
##
    bbox =   [
    21.40498391147844,
    46.112238019247,
    21.975365148504864,
    46.504322391308165
  ]
    soil_range = [0, 0.2]
    veg_range = [0.5, 1]

    dflist = []
    months = pd.date_range(time_start,time_stop,freq="ME")
    timesteps = Data.DATACUBES_L1["SLSTR"].time

    for d in timesteps:
        try:
            dflist.append(Data.process_date(date = d,  bbox= bbox,
                                            soil_range=soil_range,
                                            veg_range=veg_range))
        except Exception as e:
            print(e)

    complete_df = pd.concat(dflist)

    # plt.close("all")
    plot_hexbin(complete_df,"soil_temp", "tsurf_ka")
    plot_hexbin(complete_df,"veg_temp", "tsurf_ka", xlim= [273,320], ylim = [273,320])

##
    date = "2024-03-01"

    Data.temperatures_dashboard(bbox=bbox,date=date, scatter_x= "veg_temp")
    Data.plot_AMSR2(bbox=bbox,date=date)


##

fig = boxplot_timeseries(complete_df)
plt.show()