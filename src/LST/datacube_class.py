from typing import  Literal, List
from LST.datacube_loader import (
    OPTICAL_datacube,
    MICROWAVE_datacube,
)
from LST.datacube_utilities import calc_Holmes_temp, crop2roi, get_edges


class DATA_READER:

    def __init__(self,
                 region: Literal["sahel", "siberia", "midwest","ceu"],
                 bbox: List[float],
                 time_start:str,
                 time_stop:str,
                 microwave_overpass: Literal["day", "daynight", "night"] = "daynight",
                 ):
        """
        Class to store Level-1 data from SLSTR and AMSR2. Stroing in a class avoids reloading every iteration.
        :param region: Literal["sahel", "siberia", "midwest","ceu"]
        :param time_start: String: start date to restrict open_mfdataset to. This avoids loading too much data.
        :param time_stop: String: end date to restrict open_mfdataset to.
        :param microwave_overpass: Select which specific overpass to load for the Microwave sensor.
        """

        self.MODIS_NDVI, self.MODIS_LST = OPTICAL_datacube(region=region,
                                                 bbox=bbox,
                                                 time_start=time_start,
                                                 time_stop=time_stop
                                                           )

        self.AMSR2_BT = MICROWAVE_datacube(bbox=bbox,
                                        overpass=microwave_overpass,
                                        time_start=time_start, time_stop=time_stop)

        self.AMSR2_LST = calc_Holmes_temp(self.AMSR2_BT)



    def match_AMSR2_extent(self):
        """
        MODIS is cut to the full spatial extent of AMSR2.
        Both AMSR2 and MODIS cropped to bbox
        This is required, as when we crop by bounding box, the pixel centers are taken into account
        This is not usually a problem for high resolution data, but the 10-25km pixels for microwave sensors,
        have siginificant areas "overhanging" outside the specified region.
        """
        res = self.AMSR2_BT.attrs["resolution"]

        AMSR2_bbox = [get_edges(self.AMSR2_BT.lon.values, res).min(),
                      get_edges(self.AMSR2_BT.lat.values, res).min(),
                      get_edges(self.AMSR2_BT.lon.values, res).max(),
                      get_edges(self.AMSR2_BT.lat.values, res).max()]

        MODIS_NDVI_c = crop2roi(self.MODIS_NDVI, AMSR2_bbox)
        MODIS_LST_c = crop2roi(self.MODIS_LST, AMSR2_bbox)

        return MODIS_NDVI_c, MODIS_LST_c



# DATACUBE PROCESSORS
def match_AMSR2_date(OPTI, AMSR2, date):
    """
    Select the closest date to SLSTR, and thus select this date to access AMSR2
    """
    OPTI = OPTI.drop_duplicates(dim="time")
    OPTI_obs = OPTI.sel(time=date, method="nearest")

    # We select OPTI's observation to get AMSR2. the frequency of obs for AMSR2 is higher.
    AMSR2_obs = AMSR2.sortby('time').sel(time=OPTI_obs.time.dt.floor("d"), method="nearest")

    return OPTI_obs, AMSR2_obs



