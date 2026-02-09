# LPRM Testbed

This repo is a test environment for the [Land Parameter Retrieval Model (LPRM)](https://www.geo.vu.nl/~jeur/lprm/) radiative transfer algorithm.

## Features

* **Brightness Temperature Validation:** Resampled brightness temperatures from the [AMSR2](https://www.earthdata.nasa.gov/data/instruments/amsr2) instrument are validated against [AMPR](https://airbornescience.nasa.gov/instrument/AMPR) (Advanced Microwave Precipitation Radiometer) from NASA ER2.
* **Surface Temperature comparison:** A comparison of surface temperatures derived from AMSR2 Ka-band observations against thermal infrared data from the [Sentinel-3 SLSTR](https://sentiwiki.copernicus.eu/web/s3-slstr-instrument)
* **Algorithm Testing:** framework for evaluating LPRM, and running forward simulations as well, to simulate brightness temps.
