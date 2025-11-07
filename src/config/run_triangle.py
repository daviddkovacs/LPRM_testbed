from readers.Sat import BTData, LPRMData
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
import pandas as pd
from utilities.utils import (bbox,
                             calc_surface_temperature,
                             mpdi,
                             find_common_coords,
                             normalize)
import mpl_scatter_density
from utilities.plotting import scatter_density
from config.paths import path_lprm, path_bt

list =   [
    -9.808900000340572,
    34.73583206243012,
    26.713989828239306,
    58.97591997737254
  ]

# Frequencies(AMSR2):
AMSR2_bands = ['6.9', '7.3', '10.7', '18.7', '23.8', '36.5', '89.0']
_path_bt = path_bt
_path_lprm = path_lprm
sat_freq = '10.7'
sat_sensor = "amsr2"
overpass = "day"
target_res = "10"

composite_start = "2024-10-01"
composite_end = "2024-10-04"

datelist = pd.date_range(start=composite_start, end=composite_end)
datelist = [s.strftime("%Y-%m-%d") for s in datelist]

ref_compound = pd.DataFrame({})
test_compound = pd.DataFrame({})


for d in datelist:

    BT = BTData(path = _path_bt,
                   date = d,
                   sat_freq = sat_freq,
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    BT = BT.to_pandas()
    BT = bbox(BT, list)

    BT["MPDI"] =  mpdi(BT["BT_V"], BT["BT_H"])


    LPRM = LPRMData(path =_path_lprm,
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
# common_data["VOD_C1_s"] = normalize(common_data["VOD_C1"])
# common_data["TSURF_s"] = normalize(common_data["TSURF"])

scatter_density(
    ref=common_data["VOD_KU"],
    test=common_data["TSURF"],
    test_colour=common_data["SM_C1"],
    xlabel= "VOD_KU",
    ylabel="TSURF",
    cbar_label= "SM_C1",
    cbar_type = "jet",
    xlim = (0,1.4),
    ylim = (270,340),
    cbar_scale = (0,0.5),
    # dpi =50
    )
