from readers.Air import AirborneData
from readers.Sat import SatelliteData
from readers.ERA5 import ERA
from utilities.utils import mpdi, collocate_datasets
from utilities.plotting import scatter_plot, longitude_plot

def main_validator(ref_obj, test_obj,  *args):

    # We load the dataframes
    air_pd = ref_obj.to_pandas()
    sat_pd = test_obj.to_pandas()

    anc_pd = args[0].to_pandas()
    anc_var = args[0].variable
    # holnap szepen nezd at ezt hogy kell kicsomagolni

    # Calc MPDI for satellite
    sat_pd["MPDI"] = mpdi(sat_pd["bt_V"], sat_pd["bt_H"])

    ref_nn, test_nn = collocate_datasets( air_pd, sat_pd)
    _, anc_nn = collocate_datasets( air_pd, anc_pd)

    # anc_nn = anc_nn /10

    # scatter_plot(ref_nn["MPDI"],
    #              test_nn["MPDI"],
    #              ref_obj,
    #              test_obj)

    longitude_plot(ref_x= air_pd["lon"] ,
                   ref_y = air_pd["MPDI"],
                   test_x = test_nn["lon"],
                   test_y = test_nn["MPDI"],
                   test2_x= anc_nn["lon"],
                   test2_y = anc_nn[anc_var],
                   air_obj = ref_obj,
                   sat_obj = test_obj,)


if __name__ == "__main__":
    """
    #### Airborne Setup ####
    Frequencies:
        '10.7', '19.35', '37.1'
    Flight Directions:
        'EW', 'WE'
    Scan directions:
        '1_25', '26_50'


    #### Satellite Setup ####
    Frequencies (AMSR2):
        '6.9', '7.3', '10.7', '18.7', '23.8', '36.5', '89.0'
    Sensor:
        "amsr2" (more to come..)
    Target resolution:
        '10', '25' (kms)
    Overpass:
        'day', 'night'


    #### Common Setup ####
    date:
        '2024-10-22', '2024-10-25', '2024-10-31'
    figpath:
        if defined, saves figs

    """
    # Configure the parameters here ====================================================================================
    # Airborne (AMPR) variables
    path_air = r"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/WHYMSIE/data_from_RichDJ"
    air_freq = "10.7"
    flight_direction = "EW"
    scan_direction = "1_25"

    # Satellite (AMSR2) variables
    path_sat = r"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/passive_input/medium_resolution/AMSR2"
    sat_freq = "10.7"
    sat_sensor = "amsr2"
    overpass = "day"
    target_res = "10"

    # ERA 5 variables
    path_era = "/home/ddkovacs/shares/climers/Datapool/ECMWF_reanalysis/01_raw/ERA5-Land/datasets/images"
    variable = "lai_lv"
    # Comomn variables
    date = "2024-10-22"
    figpath = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/WHYMSIE/figures/25km"

    #  =================================================================================================================

    ER2_flight = AirborneData(path=path_air,
                              date=date,
                              scan_direction=scan_direction,
                              flight_direction=flight_direction,
                              air_freq=air_freq,
                              )

    AMSR2_OBS = SatelliteData(path=path_sat,
                              sat_sensor=sat_sensor,
                              date=date,
                              overpass=overpass,
                              target_res=target_res,
                              sat_freq=sat_freq,
                              )
    ERA_SM  = ERA(path=path_era,
                  date =date,
                  variable=variable)


    main_validator(ER2_flight,
                   AMSR2_OBS,
                   ERA_SM)