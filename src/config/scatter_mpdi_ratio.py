import os
import glob
from config.paths import path_bt, path_lprm
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from readers.Sat import BTData, LPRMData
from scipy.stats import gaussian_kde
import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import xarray as xr
import matplotlib.pyplot as plt
from lprm.retrieval.lprm_v6_1.parameters import get_lprm_parameters_for_frequency
from lprm.satellite_specs import get_specs
import lprm.retrieval.lprm_v6_1.par100m_v6_1 as par100
from shapely.geometry import LineString,  Point
from lprm.retrieval.lprm_v6_1.run_lprmv6 import load_band_from_ds
from lprm.retrieval.lprm_general import load_aux_file
from utilities.run_lprm import run_band as run_band_py

from utilities.retrieval_helpers import (
    soil_canopy_temperatures,
    interceptor,
    dummy_line, retrieve_LPRM,tiff_df
)
# for sb in ["C1","X","KU"]:
#     for nb in ["C1","X","KU"]:
year = "2024"
sat_band = "C1"
# sat_band = sb
frequencies={'C1': 6.9, 'C2': 7.3, 'X': 10.7,'KU': 18.7, 'K': 23.8, 'KA': 36.5}
sat_sensor = "amsr2"
bbox = [
    -164.37219291310385,
    -51.95094634103142,
    183.7489370276607,
    70.75914606353328
  ]

bt_path = os.path.join(path_bt,sat_sensor.upper(),"day",f"{year}*", f"*day_{year}*.nc")
bt_files = glob.glob(bt_path)

slope_list = []
intercept_list = []
# for d in range(0,360):

bt_data = xr.open_dataset(bt_files[110], decode_timedelta=False)
bt_data = bt_data.sel(lat = slice(bbox[3],bbox[1]),
                      lon = slice(bbox[0], bbox[2]))


BTV = bt_data[f"bt_{frequencies[sat_band]}V"].isel(time = 0,drop=True)
BTH = bt_data[f"bt_{frequencies[sat_band]}H"].isel(time = 0,drop=True)
BTKaV = bt_data[f"bt_36.5V"].isel(time = 0,drop=True)

num_T  = "KU" # Normally KU band (numerator)
# num_T  =nb
denominator_T = "KA" # Normally KA band (denominator)
corrector = "H/V" # Normally H/V (Ka&vertical must go hand in hand otherwise algebra wont work!
norm_T_num = bt_data[F"bt_{frequencies[num_T]}{corrector.split("/")[0]}"].isel(time = 0,drop=True)
denom_T_num = bt_data[f"bt_{frequencies[denominator_T]}{corrector.split("/")[1]}"].isel(time = 0,drop=True)

mpdi = ((BTV - BTH) / (BTV + BTH))
kuka= norm_T_num / denom_T_num

df = pd.DataFrame({"BTV" : BTV.values.flatten(),
                  "BTH" : BTH.values.flatten(),
                  "norm_T_num" : norm_T_num.values.flatten(),
                  "denom_T_num" : denom_T_num.values.flatten(),
                   "mpdi": mpdi.values.flatten(),
                   "kuka":kuka.values.flatten()})
df = df.dropna(how="any")

#Huber Regressor fitting (linear fit penalizing outliers)
ransac = HuberRegressor()
ransac.fit(df["kuka"].values.reshape(-1, 1), df["mpdi"])
line_x_ransac = np.arange(df["kuka"].min(), df["kuka"].max(), 0.01)[:, np.newaxis]
line_y_ransac = ransac.predict(line_x_ransac)
m, c = np.polyfit(line_x_ransac.ravel(), line_y_ransac, 1)

if corrector == "H/V":
    Teff = ((0.893*norm_T_num) / (1- (mpdi/abs(c)))) + 44.8
elif corrector == "V/H":
    Teff = 0.893 * ( ((norm_T_num*mpdi)/abs(c)) + norm_T_num) + 44.8

TeffKa = BTKaV * 0.893 + 44.8


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
        ax.plot(
            line_x_ransac,
            line_y_ransac,
            color="cornflowerblue",
            linewidth=2,
            label="RANSAC regressor",
        )
        fig.suptitle(f"{sat_band}-MPDI {num_T}-Norm\n slope: {np.round(m, 2)}, intercept: {np.round(c, 2)}")

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
    plt.title(f"{sat_band}-MPDI {num_T}-Norm)")
    plt.show()
    # slope_list.append(m)
    # intercept_list.append(c)

hexbin_plot(df["kuka"].values,
            df["mpdi"].values,
            type = "log",
            xlabel = f"{num_T} H / {denominator_T} V",
            ylabel = f"mpdi_{sat_band}",
            plot_polyfit=True,
            # color_array=
            # plot_TeffKa=True
            xlim = [0.2,1.3],
            ylim = [0,0.4],
            )

# hexbin_plot(df["TeffKa"].values,
#             df["mpdi"].values,
#             type = "log",
#             xlabel = f"THolmes Ka",
#             ylabel = f"mpdi",
#             # plot_1to1=True,
#             # xlim = [270,330],
#             # ylim = [270,330],
#             # color_array=df["mpdi"].values
#             # plot_TeffKa=True
#             )
##

specs = get_specs(sat_sensor.upper())
params = get_lprm_parameters_for_frequency(sat_band, specs.incidence_angle)

def lprm_retrieval(selector):
    sm, vod= run_band_py(
        BTV.values.astype('double'),
        BTH.values.astype('double'),
        TeffKa.values.astype('double'),
        load_aux_file(0.25,"SND"),
        load_aux_file(0.25,"CLY"),
        load_aux_file(0.25,"BLD"),
        params.Q,
        params.w,
        0,
        specs.incidence_angle[0],
        params.h1,
        params.h2,
        params.vod_Av,
        params.vod_Bv,
        float(get_specs(sat_sensor.upper()).frequencies[sat_band.upper()]),
        params.temp_freeze,
        False,
        None,
        # T_theor=THolmes.values.astype('double'),
        # Theory_select = selector
    )
    sm = np.where(sm<0, np.nan, sm)
    vod = np.where(vod<0, np.nan, vod)
    return sm, vod

sm_plain, vod_plain = lprm_retrieval(0)

sm_plain = xr.DataArray(
    data=sm_plain,
    dims=Teff.dims,
    coords=Teff.coords,
    name=f'sm'
)
vod_plain = xr.DataArray(
    data=vod_plain,
    dims=Teff.dims,
    coords=Teff.coords,
    name=f'vod'
)

sm_edit, vod_edit = lprm_retrieval(1)

sm_edit = xr.DataArray(
    data=sm_edit,
    dims=Teff.dims,
    coords=Teff.coords,
    name=f'sm'
)
vod_edit = xr.DataArray(
    data=vod_edit,
    dims=Teff.dims,
    coords=Teff.coords,
    name=f'vod'
)

##
def make_interactive(fig, ax, data, label):
    x_dim, y_dim = data.dims[-1], data.dims[-2]

    def on_move(event):
        if event.inaxes != ax: return
        if event.xdata is None: return
        try:
            val = data.sel({x_dim: event.xdata, y_dim: event.ydata}, method="nearest").item()
            ax.set_title(f"{label}: {val:.2f}" if not np.isnan(val) else f"{label}: NaN")
            fig.canvas.draw_idle()
        except:
            pass

    fig.canvas.mpl_connect("motion_notify_event", on_move)

fig,axs= plt.subplots(3, 2, figsize=(18, 5), constrained_layout=True)

TeffKa.plot(ax=axs[0,0], vmin=270, vmax=330, cmap="viridis")
axs[0,0].set_title("THolmes Ka")
make_interactive(fig, axs[0,0], TeffKa, "THolmes Ka")

Teff_plot = xr.where((Teff - TeffKa) <20,Teff,TeffKa)
Teff_plot.plot(ax=axs[1,0], vmin=270, vmax=330, cmap="viridis")
axs[1,0].set_title(f"THolmes ({sat_band}-MPDI {num_T}/{denominator_T})")
make_interactive(fig, axs[1,0], Teff, f"THolmes ({sat_band}-MPDI  {num_T}/{denominator_T})")

diff_t = Teff_plot - TeffKa
diff_t.plot(ax=axs[2,0], vmin=-30, vmax=30, cmap="coolwarm")
axs[2,0].set_title(f"Diff (THolmes - TeffKa)\n{sat_band}-MPDI  {num_T}/{denominator_T} T")
make_interactive(fig, axs[2,0], diff_t, f"Diff (THolmes - TeffKa)")

mpdi.plot(ax=axs[0,1], vmin=0, vmax=0.3, cmap="viridis")
axs[0,1].set_title(f"mpdi")
make_interactive(fig, axs[0,1], sm_plain, f"mpdi")

kuka.plot(ax=axs[1,1],  cmap="viridis")
axs[1,1].set_title("kuka")
make_interactive(fig, axs[1,1], sm_edit, "kuka")
plt.show()

matplotlib.use("TkAgg")
print(f"sat_band: {sat_band}")
print(f"norm_T: {num_T}")
print(f"m: {m}")
print(f"c: {c}")
