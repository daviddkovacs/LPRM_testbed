from LST.comparison_utils import (
    SLSTR_AMSR2_datacubes,
    preprocess_datacubes,
    threshold_ndvi,
    compare_temperatures,
    mpdi,
)
from plot_functions import (
    temps_plot,
    LST_plot_params,
    NDVI_plot_params,
    AMSR2_plot_params,
    plot_amsr2,combined_dashboard
)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")



if __name__=="__main__":
    DATACUBES_L1 = SLSTR_AMSR2_datacubes(region="midwest")
##
    date = "2024-02-10"

    bbox =  [
    -105.44379160961921,
    31.712595404379883,
    -100.00625973196891,
    34.644476054184466
  ]

    ndvi_threshold  = 0.5
    mpdi_band = "C1"

    DATACUBES_L2 = preprocess_datacubes(SLSTR=DATACUBES_L1["SLSTR"],
                                        AMSR2 = DATACUBES_L1["AMSR2"],
                                        date=date,
                                        bbox=bbox)

    SLSTR_LST = DATACUBES_L2["SLSTR"]["LST"]
    SLSTR_NDVI = DATACUBES_L2["SLSTR"]["NDVI"]
    AMSR2_LST = DATACUBES_L2["AMSR2"]["TSURF"]
    AMSR2_MPDI = mpdi(DATACUBES_L2["AMSR2"],mpdi_band)

    soil_temp, veg_temp = threshold_ndvi(lst = SLSTR_LST,
                                         ndvi = SLSTR_NDVI,
                                         ndvi_thres = ndvi_threshold)

    # plot_amsr2(AMSR2_LST,AMSR2_plot_params)

    df = compare_temperatures(soil_temp, veg_temp, AMSR2_LST, MPDI=AMSR2_MPDI)
    _df = df.sort_values(by="tsurf_ka")
    # temps_plot(_df)

    combined_dashboard(
        DATACUBES_L1["SLSTR"]["LST"],
        DATACUBES_L1["SLSTR"]["NDVI"],
        LST_plot_params, NDVI_plot_params, _df, bbox=bbox, plot_mpdi=False,
        plot_scatter=False
    )

