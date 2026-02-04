from LST.comparison_utils import (
    SLSTR_AMSR2_datacubes,
    preprocess_datacubes,
    threshold_ndvi,
    compare_temperatures,
    mpdi,
    get_nearest_obs
)
from plot_functions import (
    LST_plot_params,
    NDVI_plot_params,
    AMSR2_plot_params,
    plot_amsr2,combined_dashboard
)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

# Levels guide:
#     L1: All observations stacked in on xarray dataset,  Cloud, snow filtered SLSTR.
#     L1B: Observation for date selected, used for plotting whole SLSTR Tile. No spatial cropping yet.
#     L2: AMSR2 TSURF calculated, both cropped to ROI.

if __name__=="__main__":
    DATACUBES_L1 = SLSTR_AMSR2_datacubes(region="midwest")
##
    date = "2024-02-09"

    bbox =  [
    -104.97906697811854,
    31.788447796119954,
    -100.59432040067783,
    34.3466239385767
  ]

    ndvi_threshold  = 0.1
    mpdi_band = "C1"

    # get the nearest date observation for SLSTR, select this date for AMSR2
    DATACUBES_L1B = get_nearest_obs(DATACUBES_L1["SLSTR"],
                                    DATACUBES_L1["AMSR2"],
                                    date)
    # Calculating AMSR2 TSURF, cropping to bbox coords.
    DATACUBES_L2 = preprocess_datacubes(SLSTR=DATACUBES_L1B["SLSTR"],
                                        AMSR2 =DATACUBES_L1B["AMSR2"],
                                        bbox=bbox)

    SLSTR_LST = DATACUBES_L2["SLSTR"]["LST"]
    SLSTR_NDVI = DATACUBES_L2["SLSTR"]["NDVI"]
    AMSR2_LST = DATACUBES_L2["AMSR2"]["TSURF"]
    AMSR2_MPDI = mpdi(DATACUBES_L2["AMSR2"],mpdi_band)

    soil_temp, veg_temp = threshold_ndvi(lst = SLSTR_LST,
                                         ndvi = SLSTR_NDVI,
                                         soil_range =[0,0.3],
                                         ndvi_range =[0.8,1])


    # plot_amsr2(AMSR2_LST,AMSR2_plot_params)

    df = compare_temperatures(soil_temp, veg_temp, AMSR2_LST, MPDI=AMSR2_MPDI)
    _df = df.sort_values(by="tsurf_ka")

    combined_dashboard(LST_L1=DATACUBES_L1B["SLSTR"]["LST"],
                       NDVI_L1=DATACUBES_L1B["SLSTR"]["NDVI"],
                       LST_params=LST_plot_params,
                       NDVI_params=NDVI_plot_params,
                       df_S3_pixels_in_AMSR2=_df,
                       bbox=bbox, plot_mpdi=True,
                       mpdi_band = mpdi_band)

    # plt.figure()
    # plt.scatter(df["mpdi"], df["soil_mean"])
    # plt.show()