import os
import glob
from config.paths import path_bt, path_lprm
import pandas as pd
from sklearn import datasets, linear_model
from readers.Sat import BTData, LPRMData
from scipy.stats import gaussian_kde
import matplotlib
import numpy as np
matplotlib.use("Qt5Agg")
import xarray as xr
import matplotlib.pyplot as plt
from lprm.retrieval.lprm_v6_1.parameters import get_lprm_parameters_for_frequency
from lprm.satellite_specs import get_specs
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from shapely.geometry import LineString,  Point
from lprm.retrieval.lprm_v6_1.run_lprmv6 import load_band_from_ds

year = "2024"
sat_band = "C1"
frequencies={'C1': 6.9, 'C2': 7.3, 'X': 10.7,'KU': 18.7, 'K': 23.8, 'Ka': 36.5}
sat_sensor = "amsr2"
bbox = [
    15.283018365134978,
    -21.25120683446025,
    26.47320491849186,
    -10.777343049931744
  ]

bt_path = os.path.join(path_bt,"day",f"{year}*", f"*day_{year}*.nc")
bt_files = glob.glob(bt_path)

slope_list = []
intercept_list = []
# for d in range(0,360):

bt_data = xr.open_dataset(bt_files[150], decode_timedelta=False)
bt_data = bt_data.sel(lat = slice(bbox[3],bbox[1]),
                      lon = slice(bbox[0], bbox[2]))


BTV = bt_data[f"bt_{frequencies[sat_band]}V"].isel(time = 0,drop=True).values.flatten()
BTH = bt_data[f"bt_{frequencies[sat_band]}H"].isel(time = 0,drop=True).values.flatten()

KuH = bt_data["bt_6.9H"].isel(time = 0,drop=True).values.flatten()
KaV = bt_data["bt_36.5V"].isel(time = 0,drop=True).values.flatten()

df = pd.DataFrame({"BTV" : BTV,
                  "BTH" : BTH,
                  "KuH" : KuH,
                  "KaV" : KaV})

df["kuka"] = df["KuH"] / df["KaV"]
df["mpdi"] = ((df["BTV"] - df["BTH"]) / (df["BTV"] + df["BTH"]))
df["TeffKa"] = df["KaV"] *0.893 +44.8
df["Teff"] = ((0.893*df["KuH"]) / (1- (df["mpdi"]/0.58))) + 44.8
df = df.dropna(how="any")

def hexbin_plot(x, y,
                xlabel = None,
                ylabel = None,
                xlim=None,
                ylim=None,
                type = None,
                plot_TeffKa=False,
                plot_polyfit = False,
                plot_1to1 = False,
                color_array= None
                ):

    fig, ax = plt.subplots()
    hb = ax.hexbin(
        x,
        y,
        gridsize=100,
        bins= type,
        cmap = "magma",
        reduce_C_function=np.mean,
        C=color_array,
    )
    if plot_polyfit:
        ransac = linear_model.RANSACRegressor()
        ransac.fit(x.reshape(-1, 1), y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_x_ransac = np.arange(x.min(), x.max(),0.01)[:, np.newaxis]
        line_y_ransac = ransac.predict(line_x_ransac)

        ax.plot(
            line_x_ransac,
            line_y_ransac,
            color="cornflowerblue",
            linewidth=2,
            label="RANSAC regressor",
        )
        m, c = np.polyfit(line_x_ransac.ravel(), line_y_ransac,1)
        ax.plot(x, m * x + c, color='red', alpha =0.7,linestyle='--', linewidth=2)
        fig.suptitle(f"slope: {np.round(m, 2)}, intercept: {np.round(c, 2)}")

    if plot_1to1:
        ax.axline((0,0),slope=1)
    if plot_TeffKa:
        ax.hexbin(
            x,
            df["TeffKa"],
            gridsize=100,
            bins=type,
            reduce_C_function=np.mean,
            C=color_array,
        )
    plt.title(f"{sat_band} band")
    fig.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    offsets = hb.get_offsets()
    values = hb.get_array()

    def on_move(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        dx = offsets[:, 0] - event.xdata
        dy = offsets[:, 1] - event.ydata
        d2 = dx*dx + dy*dy
        i = np.argmin(d2)

        val = values[i]
        if np.ma.is_masked(val):
            ax.set_title("density = NaN")
        else:
            ax.set_title(f"density = {val:.2f}")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.show()
    # slope_list.append(m)
    # intercept_list.append(c)

hexbin_plot(df["kuka"].values,
            df["mpdi"].values,
            type = "log",
            xlabel = f"Ku H / Ka V",
            ylabel = f"mpdi_{sat_band}",
            plot_polyfit=True,
            xlim = [0.7,1.1],
            ylim = [0,0.2],
            )

# hexbin_plot(df["mpdi"].values,
#             df["Teff"].values,
#             type = "log",
#             xlabel = f"Teff Ka",
#             ylabel = f"Teff X",
#             # plot_1to1=True,
#             # xlim = [270,330],
#             # ylim = [270,330],
#             # color_array=df["mpdi"].values
#             plot_TeffKa=True
#             )

##
# plt.plot(slope_list,label = "Slope")
# plt.plot(intercept_list, label = "Intercept")
# plt.title(f"{sat_band} band\nmean slope: {np.round(np.nanmean(slope_list),2)},  mean intercept: {np.round(np.nanmean(intercept_list),2)}")
# plt.xlabel("DOY (2024)")
# plt.ylim([-0.9,0.9])
# plt.legend()
# plt.show()