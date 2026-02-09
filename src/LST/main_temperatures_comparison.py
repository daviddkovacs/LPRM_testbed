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
    )
##

    bbox = [
    21.55840069258295,
    46.09277553088265,
    21.996961171622758,
    46.3148971976035
  ]
    date = "2024-05-01"

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

    Data.temperatures_dashboard(bbox=bbox,date=date, scatter_x= "soil_temp")
    Data.plot_AMSR2(bbox=bbox,date=date)