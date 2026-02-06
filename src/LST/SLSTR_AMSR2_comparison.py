import pandas as pd
from LST.plot_functions import plot_hexbin
from LST.Datacube import SLSTR_AMSR2_DC
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")



if __name__=="__main__":

    bbox =  [
    -3.15469089911727,
    12.631478720321937,
    0.6008523636280643,
    16.30729407053441
  ]

    data = SLSTR_AMSR2_DC(
        region="sahel",
        bbox = bbox
    )
##
    dflist = []

    for m in range(1,13):
        date = f"2024-{m}-20"
        try:
            dflist.append(data.process_date(date = date))
        except Exception as e:
            print(e)

    complete_df = pd.concat(dflist)


##

plot_hexbin(complete_df,"soil_temp", "tsurf_ka")