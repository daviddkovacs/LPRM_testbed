from LST.comparison_utils import (
    SLSTR_AMSR2_datacubes,
    spatial_subset_dc,
    threshold_ndvi,
    compare_temperatures,
    mpdi,
    KuKa,
    calc_Holmes_temp,
    calc_adjusted_temp,
    temporal_subset_dc,
)
from plot_functions import (
    LST_plot_params,
    NDVI_plot_params,
    AMSR2_plot_params,
    plot_amsr2,
    combined_validation_dashboard
)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

# Levels guide:
#     L1: All observations stacked in on xarray dataset,  Cloud, snow filtered SLSTR.
#     L1B: Observation for date selected, used for plotting whole SLSTR Tile. No spatial cropping yet.
#     L2: AMSR2 TSURF calculated, both cropped to ROI.

if __name__=="__main__":
    DATACUBES_L1 = SLSTR_AMSR2_datacubes(region="sahel")
##
    date = "2024-10-20"

    bbox =  [
    -3.5530521367710435,
    13.840776476333943,
    2.7279040178611353,
    17.52977087031826
  ]
    # Soil and Vegetation masks based on NDVI
    soil_range = [0, 0.2]
    veg_range = [0.5, 1]
    mpdi_band = "x"

    # get the nearest date observation for SLSTR, select this date for AMSR2
    DATACUBES_L1B = temporal_subset_dc(SLSTR=DATACUBES_L1["SLSTR"],
                                       AMSR2=DATACUBES_L1["AMSR2"],
                                       date=date)
    # Cropping to bbox coords.
    DATACUBES_L2 = spatial_subset_dc(SLSTR=DATACUBES_L1B["SLSTR"],
                                     AMSR2=DATACUBES_L1B["AMSR2"],
                                     bbox=bbox)

    SLSTR_LST = DATACUBES_L2["SLSTR"]["LST"]
    SLSTR_NDVI = DATACUBES_L2["SLSTR"]["NDVI"]

    AMSR2_LST = calc_Holmes_temp(DATACUBES_L2["AMSR2"])
    AMSR2_LST_theor = calc_adjusted_temp(DATACUBES_L2["AMSR2"], factor= 0.8, bandH= "ku", mpdi_band=mpdi_band)
    AMSR2_MPDI = mpdi(DATACUBES_L2["AMSR2"], mpdi_band)
    AMSR2_KUKA = KuKa(DATACUBES_L2["AMSR2"], num="ku", denom="ka")

    soil_temp, veg_temp = threshold_ndvi(lst = SLSTR_LST,
                                         ndvi = SLSTR_NDVI,
                                         soil_range =soil_range,
                                         ndvi_range =veg_range)


    # plot_amsr2(AMSR2_LST,AMSR2_plot_params)

    df = compare_temperatures(soil_temp, veg_temp, AMSR2_LST, MPDI=AMSR2_MPDI, KUKA=AMSR2_KUKA, TSURFadj=AMSR2_LST_theor)
    _df = df.sort_values(by="kuka")

    combined_validation_dashboard(LST_L1=DATACUBES_L1B["SLSTR"]["LST"],
                                   NDVI_L1=DATACUBES_L1B["SLSTR"]["NDVI"],
                                   LST_params=LST_plot_params,
                                   NDVI_params=NDVI_plot_params,
                                   df_S3_pixels_in_AMSR2=_df,
                                   bbox=bbox,
                                   plot_mpdi=False,
                                   plot_kuka=True,
                                  plot_tsurf_adjust=True,
                                   mpdi_band = mpdi_band,
                                  scatter_x = 'soil_temp'
                                  )
