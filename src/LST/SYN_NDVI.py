import matplotlib.pyplot as plt
from NDVI_utils import open_sltsr, filter_empty
from config.paths import SLSTR_path
import xarray as xr

bbox = [
    -100.36610918491856,
    33.20379972639941,
    -95.04826212764668,
    37.27726050323034
]

date = "2024-05-05"

if __name__=="__main__":


    NDVI = open_sltsr(path=SLSTR_path,
                   subdir_pattern=f"S3A_SL_2_LST____*",
                   date_pattern=r'___(\d{8})T(\d{4})',
                   variable_file="LST_ancillary_ds.nc",
                   bbox= bbox
                        )

    LST= open_sltsr(path=SLSTR_path,
                   subdir_pattern=f"S3A_SL_2_LST____*",
                   date_pattern=r'___(\d{8})T(\d{4})',
                   variable_file="LST_in.nc",
                   bbox= bbox
                        )

    _SLSTR = xr.merge([NDVI,LST])[["LST","NDVI"]]

    SLSTR = filter_empty(_SLSTR,"NDVI")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

SLSTR["LST"].sel(time=date, method="nearest").plot(
    x="lon",
    y="lat",
    ax=ax1,
    cmap="coolwarm",
    cbar_kwargs={'label': 'LST [K]'}
)
ax1.set_title(f"LST")

SLSTR["NDVI"].sel(time=date, method="nearest").plot(
    x="lon", y="lat", ax=ax2, cmap="YlGn", robust=True
)
ax2.set_title(f"NDVI\n{date}")
plt.suptitle(f"Sentinel-3 SLSTR {date}")
plt.show(block=True)
