from typing import  Literal, List
from LST.comparison_utils import (
    SLSTR_AMSR2_datacubes,
    spatial_subset_dc,
    threshold_ndvi,
    compare_temperatures,
    mpdi,
    KuKa,
    calc_Holmes_temp,
    calc_adjusted_temp,
    temporal_subset_dc,
)
from plot_functions import (combined_validation_dashboard,
                            LST_plot_params,
                            AMSR2_plot_params,
                            NDVI_plot_params,
                            amsr2_lst_figure)



class SLSTR_AMSR2_DC:

    def __init__(self,
                 region:Literal["sahel", "siberia", "midwest","ceu"],
                 ):
        """
        Class to store Level-1 data from SLSTR and AMSR2. Stroing in a class avoids reloading every iteration.
        :param region: Literal["sahel", "siberia", "midwest","ceu"]
        """

        self.DATACUBES_L1 = SLSTR_AMSR2_datacubes(region=region)
        print("Data loaded.")

        self.DATACUBES_L1B = None
        self.DATACUBES_L2 = None


    def spatio_temporal_subset(self,
                               bbox: List[float],
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
        self.DATACUBES_L1B = temporal_subset_dc(
            SLSTR=self.DATACUBES_L1["SLSTR"],
            AMSR2=self.DATACUBES_L1["AMSR2"],
            date=date)

        # Cropping to bbox coords.
        self.DATACUBES_L2 = spatial_subset_dc(
            SLSTR=self.DATACUBES_L1B["SLSTR"],
            AMSR2=self.DATACUBES_L1B["AMSR2"],
            bbox=bbox)


    def process_date(self,
                     bbox,
                     date,
                     soil_range=[0, 0.2],
                     veg_range=[0.5, 1],
                     mpdi_band="x",
                     ):
        """
        Processes Soil and Vegetation temperatures for a date, and compares it to overlying AMSR2 pixels.

        :param date: Date
        :param bbox: List["lonmin", "latmin", "lonmax", "latmax"]
        :param soil_range: NDVI range in which SLSTR pixel is considered as soil.
        :param veg_range: NDVI range in which SLSTR pixel is considered as vegetation.
        :param mpdi_band: IEEE nomenclature band to calculate the Microwave Polarisation Difference Index
        :return: pd.Dataframe containing soil, veg. temperatures as well as AMSR2 retrievals.
        """

        self.spatio_temporal_subset(bbox,date)

        SLSTR_LST = self.DATACUBES_L2["SLSTR"]["LST"]
        SLSTR_NDVI = self.DATACUBES_L2["SLSTR"]["NDVI"]

        AMSR2_LST = calc_Holmes_temp(self.DATACUBES_L2["AMSR2"])
        AMSR2_LST_theor = calc_adjusted_temp(self.DATACUBES_L2["AMSR2"], factor= 0.8, bandH= "ku", mpdi_band=mpdi_band)
        AMSR2_MPDI = mpdi(self.DATACUBES_L2["AMSR2"], mpdi_band)
        AMSR2_KUKA = KuKa(self.DATACUBES_L2["AMSR2"], num="ku", denom="ka")

        soil_temp, veg_temp = threshold_ndvi(lst=SLSTR_LST,
                                             ndvi=SLSTR_NDVI,
                                             soil_range=soil_range,
                                             ndvi_range=veg_range)



        df = compare_temperatures(soil_temp, veg_temp, AMSR2_LST, MPDI=AMSR2_MPDI, KUKA=AMSR2_KUKA,
                                  TSURFadj=AMSR2_LST_theor
                                  )
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

        df = self.process_date(bbox,date)

        combined_validation_dashboard(LST_L1B=self.DATACUBES_L1B["SLSTR"]["LST"],
                                      NDVI_L1B=self.DATACUBES_L1B["SLSTR"]["NDVI"],
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

        self.spatio_temporal_subset(bbox,date)
        amsr2_lst=  calc_Holmes_temp(self.DATACUBES_L2["AMSR2"])
        amsr2_lst_figure(amsr2_lst,
                         AMSR2_plot_params)