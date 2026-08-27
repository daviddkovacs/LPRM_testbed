import os
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob

def double_world_plot(data1, data2, title1, title2, suptitle, cmap, cbar_range,
                      cbar_label,):

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 200,
    })

    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1, 0.05],
        hspace=0.28,
        wspace=0.06,
        top=0.90,
        bottom=0.12,
        left=0.06,
        right=0.96
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

    plot_kwargs = {
        "cmap":cmap,
        "vmin":cbar_range[0],
        "vmax": cbar_range[1],
        "add_colorbar": False,
        "rasterized": True,
    }

    mesh1 = data1.plot(ax=ax1, **plot_kwargs)
    ax1.set_title(title1, fontweight="bold", pad=8)
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    ax1.set_ylim(-60, 85)

    mesh2 = data2.plot(ax=ax2, **plot_kwargs)
    ax2.set_title(title2, fontweight="bold", pad=8)
    ax2.set_xlabel("Longitude (°)")
    ax2.set_ylabel("")
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_ylim(-60, 85)

    # Centered colorbar across both subplots (narrowed using a nested GridSpec)
    cbar_gs = gs[1, :].subgridspec(1, 3, width_ratios=[0.25, 0.5, 0.25])
    cax = fig.add_subplot(cbar_gs[0, 1])

    cbar = fig.colorbar(
        mesh2,
        cax=cax,
        orientation="horizontal",
        extend="both",
    )
    cbar.set_label(cbar_label, fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(suptitle, fontweight="bold", y=0.98)

    plt.show()