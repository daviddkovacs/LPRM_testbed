import copy
from qa4sm_reader.custom_user_plot_generator import CustomPlotObject
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from LST.test_lprm_day import load_TB_daily, date_pattern_lut,file_pattern_lut
from LST.datacube_utilities import calc_Holmes_temp
from LST.datacube_utilities import crop2roi
import matplotlib.ticker as mticker

amsr2_path = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/01_resampled_bt/coarse_resolution/AMSR2/"
zoomin_bbox =[
-11.177304921271343,
35.4538346353382,
33.80649407930892,
58.85815315416707
]
TB_DAY = xr.open_dataset(os.path.join(amsr2_path,"day/202405/amsr2_l1bt_day_20240501_25km.nc",),decode_timedelta=False).isel(time=0)
TB_NIGHT = xr.open_dataset(os.path.join(amsr2_path,"night/202405/amsr2_l1bt_night_20240501_25km.nc"),decode_timedelta=False).isel(time=0)

MPDI_DAY = (TB_DAY["bt_6.9V"] - TB_DAY["bt_6.9H"]) / (TB_DAY["bt_6.9V"] + TB_DAY["bt_6.9H"])
MPDI_NIGHT = (TB_NIGHT["bt_6.9V"] - TB_NIGHT["bt_6.9H"]) / (TB_NIGHT["bt_6.9V"] + TB_NIGHT["bt_6.9H"])

MPDI_DAY_ROI = crop2roi(MPDI_DAY, zoomin_bbox)
MPDI_NIGHT_ROI = crop2roi(MPDI_NIGHT, zoomin_bbox)
MPDI_dif = MPDI_DAY_ROI - MPDI_NIGHT_ROI
MPDI_same =  xr.where(MPDI_dif <0.0001, True, False )
binary_cmap = ListedColormap(['white', 'darkgreen'])


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5),
                         subplot_kw={'projection': ccrs.PlateCarree()})
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,  # Updated to match the 15pt general font size
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})
vmin = 0
vmax = 0.02

# --- Plot 1: Night ---
ax1 = axes[0]
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)

im = MPDI_NIGHT_ROI.plot.pcolormesh(
    ax=ax1,
    transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=vmin, vmax=vmax,
    cmap='viridis',
    add_colorbar=False
)

ax1.set_title("MPDI Night", pad=12, fontsize=15, family='serif')

# --- Plot 2: Day ---
ax2 = axes[1]
ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)

MPDI_DAY_ROI.plot.pcolormesh(
    ax=ax2,
    transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=vmin, vmax=vmax,
    cmap='viridis',
    add_colorbar=False
)

ax2.set_title("MPDI Day", pad=12, fontsize=15, family='serif')

# --- Plot 3: Same ---
ax3 = axes[2]
ax3.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax3.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)

im_same = MPDI_same.plot.pcolormesh(
    ax=ax3,
    transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=0, vmax=1,
    cmap=binary_cmap,
    add_colorbar=False
)

ax3.set_title(
    "Difference between:\nMPDI Night and MPDI Day",
    pad=12, fontsize=15, family='serif'
)

# Leave more space above the maps
fig.subplots_adjust(
    left=0.05,
    right=0.98,
    top=0.82,  # Slightly adjusted down to accommodate multi-line title cleanly
    bottom=0.25,
    wspace=0.05
)

# 2. Add Independent Shared Colorbar for ax1 and ax2
cbar_ax = fig.add_axes([0.18, 0.15, 0.30, 0.04]) # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='MPDI', ticks=[0, 0.01, 0.02])

# 3. Add Legend for ax3 underneath the plot
color_0 = im_same.cmap(im_same.norm(0))
color_1 = im_same.cmap(im_same.norm(1))

patch_0 = mpatches.Patch(facecolor=color_0, edgecolor='black', label='Not equal')
patch_1 = mpatches.Patch(facecolor=color_1, edgecolor='black', label='Equal')

ax3.legend(handles=[patch_1, patch_0],
           loc='upper center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=2,
           fontsize=18,
           handlelength=1.5,
           handleheight=1.5,
           borderpad=0.45,
           columnspacing=1.5,
           framealpha=0.9)

plt.show()