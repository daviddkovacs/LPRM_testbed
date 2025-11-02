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
from matplotlib.colors import LinearSegmentedColormap


white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def tri(test,ref,):

    mask = np.isfinite(ref) & np.isfinite(test)
    ref = ref[mask]
    test = test[mask]

    plt.figure(figsize=(6, 6))
    xy = np.vstack([ref, test])
    z = gaussian_kde(xy)(xy)

    plt.figure(figsize=(6, 6))

    plt.scatter(ref, test, c=z, s=20, cmap='viridis', )
    plt.xlim([0,2])
    plt.ylim([270,350])
    plt.show()

def scatter_density(ref,test):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    ax.scatter_density(ref, test)
    density = ax.scatter_density(ref, test,cmap=white_viridis, dpi=30)
    fig.colorbar(density, label='Number of points per pixel')
    # ax.set_xlim(0, 1.5)
    # ax.set_ylim(270, 330)
    fig.canvas.draw_idle()
    plt.pause(0.001)
    return fig, ax, density



list =  [
    -160.3937675108309,
    -55.92594055290717,
    179.48854267825544,
    70.6033586301435
  ]


path_sat = r"G:\My Drive\Munka\CLIMERS\ER2_validation\AMSR2\passive_input\medium_resolution"
sat_freq = "10.7"
sat_sensor = "amsr2"
overpass = "day"
target_res = "10"


datelist = [
    # "2023-08-28",
    # "2023-08-29",
    # "2023-08-30",
    "2024-10-24",
            "2024-10-25",
            "2024-10-26",
            "2024-10-27",]

for d in datelist:
    KA = BTData(path = path_sat,
                   date = d,
                   sat_freq = sat_freq,
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    dfKA = KA.to_pandas()
    dfKA = bbox(dfKA, list)
    dfKA["TSURF"] = calc_surface_temperature(dfKA["bt_V"])

    BT = BTData(path = path_sat,
                   date = d,
                   sat_freq = '18.7',
                   overpass = overpass,
                   sat_sensor = sat_sensor,
                   target_res = target_res,
                   )

    BT = BT.to_pandas()
    BT = bbox(BT, list)

    BT["MPDI"] =  mpdi(BT["bt_V"], BT["bt_H"])

    plt.ion()
    scatter_density(BT["MPDI"],
                    dfKA["TSURF"],
                    )


# LPRM = LPRMData(path = r"G:\My Drive\Munka\CLIMERS\ER2_validation\AMSR2\lprm_output\medium_resolution",
#                date = d,
#                sat_freq = sat_freq,
#                overpass = overpass,
#                sat_sensor = sat_sensor,
#                target_res = target_res,
#                )
#
# LPRM = LPRM.to_pandas()
# LPRM = bbox(LPRM, list)