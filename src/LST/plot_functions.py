import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
matplotlib.use("TkAgg")

LST_plot_params = {"x": "lon",
                   "y": "lat",
                   "cmap": "coolwarm",
                   "cbar_kwargs": {'label': 'LST [K]'},
                   "vmin": 273,
                   "title": "LST"
                   }
NDVI_plot_params = {
    "x": "lon",
    "y": "lat",
    "cmap": "YlGn",
    "cbar_kwargs": {'label': "NDVI [-]"},
    "vmin": 0,
    "vmax": 0.6,
    "title": "NDVI"
}
AMSR2_plot_params = {
    "cmap": "coolwarm",
    "cbar_kwargs": {'label': 'LST [K]'},
    "vmin": 290,
    "vmax": 320,
}


def amsr2_lst_figure(ds,
               plot_params):

    lon_min = np.min(ds.lon.values)
    lon_max = np.max(ds.lon.values)
    lat_min = np.min(ds.lat.values)
    lat_max = np.max(ds.lat.values)

    res = ds.attrs.get("resolution", 0.25)
    extent = [lon_min - res / 2, lon_max + res / 2, lat_min - res / 2, lat_max + res / 2]

    plt.figure()

    ds.plot.imshow(
        cmap=plot_params["cmap"],
        vmin=plot_params["vmin"],
        vmax=plot_params["vmax"],
        extent=extent
    )

    date_str = ds.time.dt.strftime('%Y-%m-%d').item()
    plt.title(f"AMSR2 LST in bounding box\n{date_str}")
    plt.show()


def usual_stats(x,y):
    r =x.corr(y)
    bias = (y - x).mean()
    rmse = np.sqrt(((y - x) ** 2).mean())
    return {"r" : r , "bias" : bias , "rmse" : rmse}



def plot_hexbin(df, x_col, y_col, xlim = [273, 325], ylim=[273, 325], plot_polyfit = True):

    x = df[x_col]
    y = df[y_col]
    stats = usual_stats(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))

    hb = ax.hexbin(x, y,
                   gridsize=100, cmap='inferno', mincnt=1)

    ax.plot(xlim, ylim, 'k--', alpha=0.8, linewidth=1, zorder=10)



    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}')

    clean_df = df[[x_col, y_col]].dropna()
    x = clean_df[x_col].values
    y = clean_df[y_col].values
    ransac = HuberRegressor()
    ransac.fit(x.reshape(-1, 1), y)
    line_x_ransac = np.arange(x.min(), x.max(), 0.01)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_x_ransac)
    m, c = np.polyfit(line_x_ransac.ravel(), line_y_ransac, 1)

    if plot_polyfit:
        ax.plot(
            line_x_ransac,
            line_y_ransac,
            color="cornflowerblue",
            linewidth=2,
            label="RANSAC regressor",
        )
        fig.suptitle(f"slope: {np.round(m, 2)}, intercept: {np.round(c, 2)}")

    textstr = '\n'.join((
        f'$R = {stats["r"]:.2f}$',
        f'$RMSE = {stats["rmse"]:.2f}$ K',
        f'$Bias = {stats["bias"]:.2f}$ K',
        f'$N = {len(x)}$',
        f"y(x)={np.round(m, 2)}x+{np.round(c, 2)}"
    ))

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    plt.show()


def fill_plot_coords(ds_slice):
    """Forward and backward fills lat/lon NaNs to prevent pcolormesh errors."""
    for coord in ["lat", "lon"]:
        ds_slice[coord] = (ds_slice[coord]
                           .ffill(dim="column").bfill(dim="column")
                           .ffill(dim="row").bfill(dim="row"))
    return ds_slice


def plot_modis_comparison(ndvi_da, lst_da, ndvi_time=4, lst_time=8):
    """Slices, fixes coordinates, and plots NDVI and LST side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ndvi_slice = fill_plot_coords(ndvi_da.sel(time=ndvi_time, method="nearest").copy())
    ndvi_slice.plot(ax=ax1, cmap="RdYlGn", x="lon", y="lat")
    ax1.set_title(f"MODIS NDVI (Time Index: {ndvi_slice.time.values})")

    lst_slice = fill_plot_coords(lst_da.sel(time=lst_time, method="nearest").copy())
    lst_slice.plot(ax=ax2, cmap="inferno", x="lon", y="lat")
    ax2.set_title(f"MODIS LST (Time Index: {lst_slice.time.values})")

    plt.tight_layout()
    plt.show()