import pandas as pd
from LST.plot_functions import plot_hexbin
from LST.SLSTR_AMSR2_reader import SLSTR_AMSR2_DC
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime

if __name__=="__main__":

    Data = SLSTR_AMSR2_DC(
        region="ceu",
        time_start="2024-01-01",
        time_stop="2025-01-01",
    )
##

    bbox =  [
    21.518989522702412,
    45.97810952647487,
    23.077199533734955,
    46.912737555251965
  ]

    loopstart =datetime.now()
    dflist = []
    months = pd.date_range("2024-01-01","2024-12-31",freq="ME")
    for d in months:
        try:
            print(d)
            dflist.append(Data.process_date(date = d,  bbox= bbox))
        except Exception as e:
            print(e)
    loopend = datetime.now()
    print(f"loop: {loopend - loopstart}")
    complete_df = pd.concat(dflist)

    plt.close("all")
    plot_hexbin(complete_df,"soil_temp", "tsurf_ka")
    plot_hexbin(complete_df,"veg_temp", "tsurf_ka")

##
    date = "2024-06-01"

    Data.temperatures_dashboard(bbox=bbox,date=date, scatter_x= "soil_temp")
    Data.plot_AMSR2(bbox=bbox,date=date)