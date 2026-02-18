from typing import  Literal, List
from LST.datacube_loader import (
    OPTICAL_datacube,
    spatial_subset_dc,
    match_OPTI_to_AMSR2_date, MICROWAVE_datacube,
)
from LST.datacube_utilities import calc_Holmes_temp, calc_adjusted_temp, KuKa, mpdi, threshold_ndvi, \
    compare_temperatures
from plot_functions import (combined_validation_dashboard,
                            LST_plot_params,
                            AMSR2_plot_params,
                            NDVI_plot_params,
                            amsr2_lst_figure)


class DATA_READER:

    def __init__(self,
                 region: Literal["sahel", "siberia", "midwest","ceu"],
                 bbox: List[float],
                 sensor: Literal["MODIS","SLSTR"],
                 time_start:str,
                 time_stop:str,
                 ):
        """
        Class to store Level-1 data from SLSTR and AMSR2. Stroing in a class avoids reloading every iteration.
        :param region: Literal["sahel", "siberia", "midwest","ceu"]
        :param time_start: String: start date to restrict open_mfdataset to. This avoids loading too much data.
        :param time_stop: String: end date to restrict open_mfdataset to.
        """

        self.MODIS_NDVI, self.MODIS_LST = OPTICAL_datacube(region=region,
                                                 sensor=sensor,
                                                 bbox=bbox,
                                                 time_start=time_start,
                                                 time_stop=time_stop
                                                           )

        self.AMSR2 = MICROWAVE_datacube(bbox=bbox,
                                        overpass="daynight",
                                        time_start=time_start, time_stop=time_stop)


    def temporal_subset(self,
                        date):
        """
        Levels guide:
            L1: All observations stacked in on xarray dataset. Instantiated by class. Cloud, snow filtered SLSTR.
            L1B: Observation for date selected, used for plotting whole SLSTR Tile. No spatial cropping yet.
            L2: AMSR2 TSURF calculated, both cropped to ROI.
        Gets the spatial subset by bbox, and temporal match as the closest observation available.
        :param bbox: List["lonmin", "latmin", "lonmax", "latmax"]
        :param date: Date in string format: "2024-10-01"
        :return: L2 Datacube of matching SLSTR and AMSR2 observations
        """

        # Selecting closest time of observation
        LST_temp, AMSR2_temp = match_OPTI_to_AMSR2_date(OPTI=self.MODIS_LST, AMSR2=self.AMSR2, date=date)

        NDVI_temp, _ = match_OPTI_to_AMSR2_date(OPTI=self.MODIS_NDVI, AMSR2=self.AMSR2, date=date)

        return LST_temp,NDVI_temp,AMSR2_temp


    def spatial_temporal_subset(self, bbox,date):

        LST_temp, NDVI_temp, AMSR2_temp = self.temporal_subset(date)

        # Cropping closest time of observation to AMSR2 extents
        LST_crop, AMSR2_crop = spatial_subset_dc(
            OPTI=LST_temp,
            AMSR2=AMSR2_temp,
            bbox=bbox)

        NDVI_crop, _ = spatial_subset_dc(
            OPTI=NDVI_temp,
            AMSR2=AMSR2_temp,
            bbox=bbox)

        return LST_crop,NDVI_crop,AMSR2_crop


    def process_date(self,
                     bbox,
                     date,
                     soil_range=[0, 0.2],
                     veg_range=[0.5, 1],
                     mpdi_band="x",
                     ):
        """
        Processes Soil and Vegetation temperatures for a date, and compares it to overlying AMSR2 pixels.

        :param bbox: List["lonmin", "latmin", "lonmax", "latmax"]
        :param date: Date
        :param soil_range: NDVI range in which SLSTR pixel is considered as soil.
        :param veg_range: NDVI range in which SLSTR pixel is considered as vegetation.
        :param mpdi_band: IEEE nomenclature band to calculate the Microwave Polarisation Difference Index
        :return: pd.Dataframe containing soil, veg. temperatures as well as AMSR2 retrievals.
        """

        OPTI_LST, OPTI_NDVI, AMSR2_BT = self.spatial_temporal_subset(bbox,date)

        AMSR2_LST = calc_Holmes_temp(AMSR2_BT)

        AMSR2_LST_theor = calc_adjusted_temp(AMSR2_BT, factor= 0.8, bandH= "ku", mpdi_band=mpdi_band)
        AMSR2_MPDI = mpdi(AMSR2_BT, mpdi_band)
        AMSR2_KUKA = KuKa(AMSR2_BT, num="ku", denom="ka")

        soil_temp, veg_temp = threshold_ndvi(lst=OPTI_LST,
                                             ndvi=OPTI_NDVI,
                                             soil_range=soil_range,
                                             ndvi_range=veg_range)



        df = compare_temperatures(soil_temp,
                                  veg_temp,
                                  AMSR2_LST,
                                  MPDI=AMSR2_MPDI,
                                  KUKA=AMSR2_KUKA,
                                  TSURFadj=AMSR2_LST_theor
                                  )
        df["time"] = soil_temp.time.values
        _df = df.sort_values(by="kuka")

        return _df


    def temperatures_dashboard(self,
                               bbox,
                               date,
                               plot_mpdi=False,
                               plot_tsurf_adjust=False,
                               plot_kuka=False,
                               mpdi_band=None,
                               scatter_x=None,
                               LST_params=LST_plot_params,
                               NDVI_params=NDVI_plot_params,
                               ):
        """
        Creates a dashboard style figure, with 1: SLSTR LST, 2: SLSTR NDVI and bbox within.
        A plot is also created which shows the SLSTR soil and vegetation temperatures per AMSR2 pixel.
        Two scatter plots show the relationship b/w AMSR2 LST and theretical LST calculated from the MPDI adjusted
        formula.
        """
        df = self.process_date(bbox,date)

        combined_validation_dashboard(LST_L1B=self.LST_temp,
                                      NDVI_L1B=self.NDVI_temp,
                                      df_S3_pixels_in_AMSR2=df,
                                      bbox=bbox,
                                      plot_mpdi=plot_mpdi,
                                      plot_tsurf_adjust=plot_tsurf_adjust,
                                      plot_kuka=plot_kuka,
                                      mpdi_band=mpdi_band,
                                      scatter_x=scatter_x,
                                      LST_params=LST_params,
                                      NDVI_params=NDVI_params,
                                      )

    def plot_AMSR2(self,bbox,date):
        """
        Plot AMSR2 Ka-band LST, within bounding box. This function allows to check, how coarse its resolution is
        as compared to SLSTR.
        :param bbox: List["lonmin", "latmin", "lonmax", "latmax"]
        :param date: Date
        :return:
        """
        self.spatio_temporal_subset(bbox,date)
        amsr2_lst = calc_Holmes_temp(self.AMSR2_crop)
        amsr2_lst_figure(amsr2_lst, AMSR2_plot_params)