from LST.comparison_utils import (
    SLSTR_AMSR2_datacubes,
    preprocess_datacubes,
    threshold_ndvi,
    compare_temperatures,
    mpdi,
)
from plot_functions import (
    temps_plot,
    AMSR2_plot_params,
    plot_amsr2,
)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")



if __name__=="__main__":
    DATACUBES = SLSTR_AMSR2_datacubes(region="siberia")
##
    date = "2024-02-10"

    bbox =  [
    114.89358681960232,
    52.344698468226056,
    122.99242187718943,
    55.12208617312004
  ]

    ndvi_threshold  = 0.5
    mpdi_band = "C1"
    DATACUBES_L2 = preprocess_datacubes(SLSTR=DATACUBES["SLSTR"],
                                        AMSR2 = DATACUBES["AMSR2"],
                                        date=date,
                                        bbox=bbox)

    SLSTR_LST = DATACUBES_L2["SLSTR"]["LST"]
    SLSTR_NDVI = DATACUBES_L2["SLSTR"]["NDVI"]
    AMSR2_LST = DATACUBES_L2["AMSR2"]["TSURF"]
    AMSR2_MPDI = mpdi(DATACUBES_L2["AMSR2"],mpdi_band)

    soil_temp, veg_temp = threshold_ndvi(lst = SLSTR_LST,
                                         ndvi = SLSTR_NDVI,
                                         ndvi_thres = ndvi_threshold)

    plot_amsr2(AMSR2_LST,AMSR2_plot_params)

    df = compare_temperatures(soil_temp,veg_temp,AMSR2_LST)
    _df = df.sort_values(by="veg_mean")
    temps_plot(_df)

