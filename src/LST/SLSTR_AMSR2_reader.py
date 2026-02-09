from typing import  Literal
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
from plot_functions import combined_validation_dashboard

# Levels guide:
#     L1: All observations stacked in on xarray dataset,  Cloud, snow filtered SLSTR.
#     L1B: Observation for date selected, used for plotting whole SLSTR Tile. No spatial cropping yet.
#     L2: AMSR2 TSURF calculated, both cropped to ROI.

class SLSTR_AMSR2_DC:

    def __init__(self,
                 region:Literal["sahel", "siberia", "midwest","ceu"],
                 ):
        """
        Class to store Level-1 data from SLSTR and AMSR2. Stroing in a class avoids reloading every iteration.
        :param region: Literal["sahel", "siberia", "midwest","ceu"]
        """

        self.DATACUBES_L1 = SLSTR_AMSR2_datacubes(region=region)
        print("loading finished")


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

        DATACUBES_L1B = temporal_subset_dc(SLSTR=self.DATACUBES_L1["SLSTR"],
                                           AMSR2=self.DATACUBES_L1["AMSR2"],
                                           date=date)
        # Cropping to bbox coords.
        DATACUBES_L2 = spatial_subset_dc(SLSTR=DATACUBES_L1B["SLSTR"],
                                         AMSR2=DATACUBES_L1B["AMSR2"],
                                         bbox=bbox)

        SLSTR_LST = DATACUBES_L2["SLSTR"]["LST"]
        SLSTR_NDVI = DATACUBES_L2["SLSTR"]["NDVI"]

        AMSR2_LST = calc_Holmes_temp(DATACUBES_L2["AMSR2"])
        # AMSR2_LST_theor = calc_adjusted_temp(DATACUBES_L2["AMSR2"], factor= 0.8, bandH= "ku", mpdi_band=mpdi_band)
        AMSR2_MPDI = mpdi(DATACUBES_L2["AMSR2"], mpdi_band)
        AMSR2_KUKA = KuKa(DATACUBES_L2["AMSR2"], num="ku", denom="ka")

        soil_temp, veg_temp = threshold_ndvi(lst=SLSTR_LST,
                                             ndvi=SLSTR_NDVI,
                                             soil_range=soil_range,
                                             ndvi_range=veg_range)

        # plot_amsr2(AMSR2_LST,AMSR2_plot_params)

        df = compare_temperatures(soil_temp, veg_temp, AMSR2_LST, MPDI=AMSR2_MPDI, KUKA=AMSR2_KUKA,
                                  # TSURFadj=AMSR2_LST_theor
                                  )
        _df = df.sort_values(by="kuka")

        return _df


    # def temperatures_dashboard(self,
    #                            bbox,
    #                            date):
    #
    #     df = self.process_date(bbox,date)
    #     combined_validation_dashboard()
    #
