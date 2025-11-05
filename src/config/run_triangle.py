from readers.Sat import BTData, LPRMData
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
import pandas as pd
from utilities.utils import (bbox,
                             calc_surface_temperature,
                             mpdi,
                             find_common_coords,
                             normalize)
import mpl_scatter_density # adds projection='scatter_density'
from utilities.plotting import scatter_density

list = [
    14.284295810477516,
    61.27841143689383,
    16.202652562513663,
    62.30595374553576
  ]

# Frequencies(AMSR2):
# '6.9', '7.3', '10.7', '18.7', '23.8', '36.5', '89.0'
path_sat = r"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/passive_input/medium_resolution/AMSR2"
sat_freq = '10.7'
sat_sensor = "amsr2"
overpass = "day"
target_res = "10"

composite_start = "2024-05-01"
composite_end = "2024-07-01"

datelist = pd.date_range(start=composite_start, end=composite_end)
datelist = [s.strftime("%Y-%m-%d") for s in datelist]

ref_compound = pd.DataFrame({})
test_compound = pd.DataFrame({})


for d in datelist:
    KA = BTData(path = path_sat,
                   date = d,
                   sat_freq = '36.5',
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    dfKA = KA.to_pandas()
    dfKA = bbox(dfKA, list)
    dfKA["TSURF"] = calc_surface_temperature(dfKA["BT_V"])

    BT = BTData(path = path_sat,
                   date = d,
                   sat_freq = sat_freq,
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    BT = BT.to_pandas()
    BT = bbox(BT, list)

    BT["MPDI"] =  mpdi(BT["BT_V"], BT["BT_H"])


    LPRM = LPRMData(path = r"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/lprm_output/medium_resolution/AMSR2",
                   date = d,
                   sat_freq = sat_freq,
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    LPRM = LPRM.to_pandas()
    LPRM = bbox(LPRM, list)

    ref_compound = pd.concat([ref_compound,LPRM])
    test_compound = pd.concat([test_compound,BT])
    print(f"{d} read")

plt.ion()
common_data = find_common_coords(ref_compound,test_compound)

common_data["VOD_C1_s"] = normalize(common_data["VOD_C1"])
common_data["TSURF_s"] = normalize(common_data["TSURF"])

scatter_density(
    ref=common_data["VOD_C1"],
    test=common_data["TSURF"],
    test_colour=common_data["SM_C1"],
    xlabel= "VOD_C1",
    ylabel="TSURF",
    cbar_label= "SM_C1",
    cbar_type = "jet",
    xlim = (0,1.4),
    ylim = (270,340),
    cbar_scale = (0,0.5),
    )
