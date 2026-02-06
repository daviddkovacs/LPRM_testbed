import pandas as pd
from LST.plot_functions import plot_hexbin
from LST.Datacube import SLSTR_AMSR2_DC
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from joblib import Parallel, delayed
from datetime import datetime

if __name__=="__main__":

    bbox =  [
    -104.1156784678568,
    32.842490923359264,
    -103.41811943484528,
    33.424464486786974
  ]

    data = SLSTR_AMSR2_DC(
        region="midwest",
    )
##
    # loopstart =datetime.now()
    # dflist = []
    # months = pd.date_range("2024-01-01","2024-12-31",freq="ME")
    # for d in months:
    #     try:
    #         print(d)
    #         dflist.append(data.process_date(date = d))
    #     except Exception as e:
    #         print(e)
    # loopend = datetime.now()
    # print(f"loop: {loopend - loopstart}")
    # complete_df = pd.concat(dflist)


##
    parstart = datetime.now()
    def worker(date_str):
        try:
            print(f"Processing {date_str}...")
            return data.process_date(date=date_str, bbox = bbox)
        except Exception as e:
            print(f"Failed on {date_str}: {e}")
            return None


    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(worker)(f"2024-{m}-20") for m in range(1, 13)
    )
    parend = datetime.now()
    print(f"parralel: {parend - parstart}")
    valid_dfs = [res for res in results if res is not None]
    complete_df = pd.concat(valid_dfs)

##

    plot_hexbin(complete_df,"soil_temp", "tsurf_ka")
    plot_hexbin(complete_df,"veg_temp", "tsurf_ka")