import pandas as pd
from LST.plot_functions import plot_hexbin
from LST.SLSTR_AMSR2_reader import SLSTR_AMSR2_DC
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from datetime import datetime

if __name__=="__main__":

    data = SLSTR_AMSR2_DC(
        region="sahel",
    )
##

    bbox =  [
    -0.10614125605815161,
    10.349338173714429,
    1.7501605707841748,
    11.840544007167708
  ]
    loopstart =datetime.now()
    dflist = []
    months = pd.date_range("2024-01-01","2024-12-31",freq="ME")
    for d in months:
        try:
            print(d)
            dflist.append(data.process_date(date = d,  bbox= bbox))
        except Exception as e:
            print(e)
    loopend = datetime.now()
    print(f"loop: {loopend - loopstart}")
    complete_df = pd.concat(dflist)

    plt.close("all")
    plot_hexbin(complete_df,"soil_temp", "tsurf_ka")
    plot_hexbin(complete_df,"veg_temp", "tsurf_ka")

##

    # parstart = datetime.now()
    # def worker(date_str):
    #     try:
    #         print(f"Processing {date_str}...")
    #         return data.process_date(date=date_str, bbox = bbox)
    #     except Exception as e:
    #         print(f"Failed on {date_str}: {e}")
    #         return None
    #
    # results = Parallel(n_jobs=-1, backend="loky")(
    #     delayed(worker)(f"2024-{m}-01") for m in range(1, 13)
    # )
    # parend = datetime.now()
    # print(f"parralel: {parend - parstart}")
    # valid_dfs = [res for res in results if res is not None]
    # complete_df = pd.concat(valid_dfs)

##
