# -*- coding: utf-8 -*-

# external packages

import math

import numpy as np
import scipy.misc

# this project
def radiative_transfer(sm,
             vod,
             Temperature,
             sand,
             clay,
             bulk_density,
             Q,
             single_scat_a,
             opt_atm,
             u,
             h1,
             h2,
             Av,
             Bv,
             f,
             temp_freeze,
             vegetation_correction = False,
             VODN = None,
             slope_mpdi = 1.0,
             intercept_mpdi = 0.0,
             return_other_stats=('mpdi', 'Teff'),
             return_other_iter_stats=tuple(),
    ):
    """
    Compute LPRM for a band.

    Parameters
    ----------
    Vertical_polarized_BT: numpy.ndarray
        Brightness temperatures of the band in V polarization
    Horizontal_polarized_BT: numpy.ndarray
        Brightness temperatures of the band in V polarization
    Temperature: numpy.ndarray
        Effective temperature
    sand: numpy.ndarray
        Sand fraction
    clay: numpy.ndarray
        clay fraction
    bulk_density: numpy.ndarray
        bulk density
    Q: double
        Polarization mixing parameter
    single_scat_a: double
        Single scattering albedo. w in formulas
    opt_atm: double
        TBD
    u: double
        Incidence angle
    h1: double
        TBD
    h2: double
        TBD
    Av: double
        TBD
    Bv: double
        TBD
    f: double
        TBD
    temp_freeze: double
        Temperature threshold for frozen conditions
    vegetation_correction: bool
        If vegetation correction needs to be applied, False by default
    VODN: Optional[numpy.ndarray]
        VOD from run without vegetation correction, used to correct for roughness
    slope_mpdi: double
        intercalibration equation (slope) to apply to C and X band ASMRE mpdi
    intercept_mpdi: double
        intercalibration equation (intercept)  to apply to C and X band ASMRE mpdi
    return_other_stats: list[str]
        List of other statistics to return. Possible values are:
        - 'mpdi': MPDI
        - 'Teff': Effective temperature
    return_other_iter_stats: list[str]
        List of other statistics to return for each iteration. Possible values are:

    Returns
    -------
    SM: numpy.ndarray
        Soil moisture
        Values from 0 to 1 are soil moisture retrievals.

        Other values:

        :-1: mdpi < 0.0001
        :-2: no coverage
        :-3: freezing

    VOD: numpy.ndarray
        Vegetation Optical Depth
        Values from 0 to 1 are VOD retrievals

        Other values:

        :-1: mdpi < 0.0001
        :-2: no coverage
        :-3: freezing
    """
    if vegetation_correction and VODN is None:
        raise ValueError("If vegetation_correction is 'True', VODN is required and cannot be 'None' ")

    smrun = np.arange(0, 1.001, 0.001)  # 1000 range
    lenWc = len(smrun)
    indbref = np.zeros(8, dtype=np.int64)
    indb = np.zeros(8, dtype=np.int64)
    opt = np.zeros(8, dtype=np.float64)
    tres = np.zeros(8, dtype=np.float64)
    tres2 = np.zeros(8, dtype=np.float64)
    optv = np.zeros(8, dtype=np.float64)
    ereal = np.zeros(8, dtype=np.float64)
    eimag = np.zeros(8, dtype=np.float64)
    Tbreal = np.zeros(8, dtype=np.float64)
    Tbimag = np.zeros(8, dtype=np.float64)
    Tbvreal = np.zeros(8, dtype=np.float64)
    Tdiff = np.zeros(8, dtype=np.float64)
    Tbvimag = np.zeros(8, dtype=np.float64)
    e1imag = np.zeros(8, dtype=np.float64)
    e1real = np.zeros(8, dtype=np.float64)
    emissivity_h = np.zeros(8, dtype=np.float64)
    ehv = np.zeros(8, dtype=np.float64)


    # NOTE: # smrun2 is a range of SM values from 0 to 1
    smrun2 = np.zeros(lenWc+9, dtype=np.float64)
    #smrun2=np.zeros(lenWc+9, type=np.float64)
    smrun2[0] = 0.0
    smrun2[1:10] = 0.001
    smrun2[10:] = smrun[1:]

    cos_u = math.cos(math.radians(u))
    sin_u = math.sin(math.radians(u))
    lon = sm.shape[0]
    lat = sm.shape[1]
    sm_ln = np.empty((lon, lat))
    sm_ln[:] = np.nan
    vod_ln = np.empty((lon, lat))
    vod_ln[:] = np.nan


    TbV_sim =  np.empty((lon, lat))
    TbH_sim = np.empty((lon, lat))
    opt_sim = np.empty((lon, lat))
    T_soil = np.empty((lon, lat))
    T_canopy = np.empty((lon, lat))
    T_return = np.empty((lon, lat))
    # % definitions from p293 Wang and Schmugge, 1980
    # dielectric constants
    # Ice:
    eice_real = 3.2
    eice_imag = 0.1

    # Air:
    eair_real = 1.
    eair_imag = 0.

    # Rock:
    erock_real = 5.5
    erock_imag = 0.2

    # istep=10
    f = f * 1e9


    for ir in range(lon):
        for ic in range(lat):

            # indb[:] = np.int64([0, 144, 288, 432, 576, 720, 864, 1008])
            Sand = sand[ir, ic]
            Clay = clay[ir, ic]
            BulkDensity = bulk_density[ir, ic]
            sm_pixel = sm[ir, ic]
            vod_pixel = vod[ir, ic]
            T = Temperature[ir, ic]

            if T > temp_freeze and BulkDensity > 0:
                # if (TBv + TBh) > 0:
                #     mpdi_l = (TBv - TBh) / (TBv + TBh) * slope_mpdi + intercept_mpdi
                # else:
                #     mpdi_l = 0.
                # if mpdi_l > 0.0001:
                WiltingP = 0.06774 - 0.064 * Sand + 0.478 * Clay # eq(1) Wang and Schmugge,1980

                Porosity = 1. - (BulkDensity / 2.65) # eq(7) Wang and Schmugge,1980

                T_celsius = T - 273.15
                # high frequency limit of the dielectric constant of pure water
                ewater_inf = 4.9
                # relaxation time of pure water	(Stogryn, 1970)
                relax_t = 1.1109e-10 - 3.824e-12 * T_celsius + 6.938e-14 * T_celsius ** 2 - 5.096e-16 * T_celsius ** 3
                # Static dielectric constant of pure water
                ewater_stat = 88.045 - 0.4147 * T_celsius + 6.295e-4 * T_celsius ** 2 + 1.075e-5 * T_celsius ** 3
                # real and imaginary parts of the dielectric constant of pure water
                ewater_real = ewater_inf + ((ewater_stat - ewater_inf) / (1 + (relax_t * f) ** 2))
                ewater_imag = (relax_t * f * (ewater_stat - ewater_inf) / (1 + (relax_t * f) ** 2))

                # The final Wang Schmugge model
                # y is the fit parameter Fig(7) Wang and Schmugge,1980
                y = -0.57 * WiltingP + 0.481

                # Wt: transition moisture Fig(7) Wang and Schmugge,1980
                Wt = 0.49 * WiltingP + 0.165

                # % dielectric constant of the initially absorbed water (Wc <= Wt)
                ew2real = ewater_real - eice_real
                ew2imag = ewater_imag - eice_imag
                ex2real = eice_real + (ewater_real - eice_real) * y
                ex2imag = eice_imag + (ewater_imag - eice_imag) * y

                smrun2 = sm_pixel
                vart = (smrun2 / Wt) * y
                ex1real = eice_real + ew2real * vart
                ex1imag = eice_imag + ew2imag * vart
                # dielectric constant of the soil (Wc <= Wt)
                # (Wc <= Wt)
                e1real = smrun2 * ex1real + \
                             (Porosity - smrun2) * eair_real + (1 - Porosity) * erock_real
                # (Wc <= Wt)
                e1imag = smrun2 * ex1imag + \
                             (Porosity - smrun2) * eair_imag + (1 - Porosity) * erock_imag
                # dielectric constant of the initially absorbed water (Wc > Wt)
                # dielectric constant of the soil (Wc > Wt)
                ereal = Wt * ex2real + (smrun2 - Wt) * ewater_real + (
                        Porosity - smrun2) * eair_real + (1 - Porosity) * erock_real  # %(Wc > Wt)
                eimag = Wt * ex2imag + (smrun2 - Wt) * ewater_imag + (
                        Porosity - smrun2) * eair_imag + (1 - Porosity) * erock_imag  # %(Wc > Wt)
                # combine (Wc <= Wt) & (Wc > Wt)
                Wcz = smrun2
                if Wcz < Wt:
                    ereal = e1real
                    eimag = e1imag

                # k: complex dielectric constant
                #TODO WHAT IS THIS
                # if ((i3 <= 9) & (i3 >= 0)):
                #     k = 1.2+i3*0.02
                # else:
                k = np.sqrt(ereal**2+eimag**2)

                # if vegetation_correction:
                #     h = h1 * (Av + Bv * VODo - 2.0 * Av * smrun2[i3])
                #     h = max(ttt, h)
                if h2 >= 0:
                    h = h1 - smrun2 * h2
                else:
                    SM_val = h2 * Porosity * -1.
                    # New roughness formulation, goes to zero at Porosity
                    if smrun2 > (SM_val):
                        fact = 1.0 - (smrun2 - SM_val) / (Porosity - SM_val)
                    else:
                        fact = 1.0

                    h = h1 * fact
                h = max(0, h)

                #sine of k - sin2u: eq(A.1) Njoku 1999
                ss = np.sqrt(k - sin_u ** 2)

                # Fresnel equations eq(A.1,2) Njoku 1999
                Rh = ((cos_u - ss) / (cos_u + ss)) ** 2
                Rv = ((k * cos_u - ss) / (k * cos_u + ss)) ** 2

                # Rough surface emissivity eq(A.12,13) Njoku 1999
                # Since we optimize for H-pol (due to higher sensitivity to SM changes)
                # H-pol emissivity is iterated for optimization
                hu = math.exp(-h * cos_u)
                emissivity_h = 1 - ((1 - Q) * Rh + Q * Rv) * hu
                emissivity_v = 1 - ((1 - Q) * Rv + Q * Rh) * hu
                d = 0.5 * (single_scat_a / (1 - single_scat_a))

                opt = vod_pixel
                trans_v = math.exp(-opt / cos_u)

                T_s = T + 3
                T_c = T - 3
                T_soil[ir, ic] =  T_s
                T_canopy[ir, ic] =T_c
                TbH_sim[ir, ic] = T_s * emissivity_h * trans_v + (1 - single_scat_a) * T_c * (1 - trans_v) + (
                        1 - emissivity_h) * (1 - single_scat_a) * T_c * (1 - trans_v) * trans_v

                TbV_sim[ir, ic] = T_s * emissivity_v * trans_v + (1 - single_scat_a) * T_c * (1 - trans_v) + (
                        1 - emissivity_v) * (1 - single_scat_a) * T_c * (1 - trans_v) * trans_v

                mpdi =((TbV_sim[ir, ic]-TbH_sim[ir, ic])/(TbV_sim[ir, ic]+TbH_sim[ir, ic]))
                a = 0.5 * ((emissivity_v - emissivity_h) / mpdi - emissivity_v - emissivity_h)
                opt_sim[ir, ic] = max(cos_u * np.log(a * d + np.sqrt((a * d) ** 2 + a + 1)), 0)
                # T_return[ir, ic] = T
            else:
                # If temperature is no data value then we have no coverage
                TbH_sim[ir, ic] = np.nan
                TbV_sim[ir, ic] = np.nan

    return TbH_sim, TbV_sim, opt_sim
