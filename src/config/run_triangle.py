from readers.Sat import BTData, LPRMData
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
import pandas as pd
from utilities.utils import bbox, calc_surface_temperature, mpdi
import mpl_scatter_density # adds projection='scatter_density'
from utilities.plotting import scatter_density



list = [
    43.1852028411314,
    56.06920505072671,
    141.24479663827634,
    64.88951768313916
  ]

path_sat = r"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/passive_input/medium_resolution/AMSR2"
sat_freq = '18.7'
sat_sensor = "amsr2"
overpass = "day"
target_res = "10"


datelist = pd.date_range(start='1/1/2024', end='12/31/2024')
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
    dfKA["TSURF"] = calc_surface_temperature(dfKA["bt_V"])

    BT = BTData(path = path_sat,
                   date = d,
                   sat_freq = sat_freq,
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    BT = BT.to_pandas()
    BT = bbox(BT, list)

    BT["MPDI"] =  mpdi(BT["bt_V"], BT["bt_H"])


    LPRM = LPRMData(path = r"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/lprm_output/medium_resolution/AMSR2",
                   date = d,
                   sat_freq = sat_freq,
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    LPRM = LPRM.to_pandas()
    LPRM = bbox(LPRM, list)

    ref_compound = pd.concat([ref_compound,BT])
    test_compound = pd.concat([test_compound,dfKA])

plt.ion()
scatter_density(
    ref_compound["bt_V"],
    test_compound["TSURF"],
    xlabel= "18.7 GHz MPDI",
    ylabel="Ka T surface",
    )
