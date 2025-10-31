from readers.Air import AirborneData
from readers.Sat import SatelliteData
from readers.ERA5 import ERA
from main import validator, validator_all

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
    flight_direction = "WE"
    scan_direction = "26_50"

    # Satellite (AMSR2) variables
    path_sat = r"/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/LPRM/passive_input/medium_resolution/AMSR2"
    sat_freq = "10.7"
    sat_sensor = "amsr2"
    overpass = "day"
    target_res = "10"

    # ERA 5 variables
    path_era = "/home/ddkovacs/shares/climers/Datapool/ECMWF_reanalysis/01_raw/ERA5-Land/datasets/images"
    bio_var = "lai_lv"
    # Comomn variables
    date = "2024-10-22"
    figpath = "/home/ddkovacs/shares/climers/Projects/CCIplus_Soil_Moisture/07_data/WHYMSIE/figures/25km"

    single_validation = False
    compound_validation = True
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
                  bio_var=bio_var)

    if single_validation:
        validator(ER2_flight,
                  AMSR2_OBS,
                  ERA_SM)

    if compound_validation:
        validator_all(path_air,
                      path_sat,
                      path_era,
                      sat_sensor=sat_sensor,
                      overpass=overpass,
                      target_res=target_res,
                      sat_freq=sat_freq,
                      air_freq=air_freq,
                      bio_var=bio_var,
                      )