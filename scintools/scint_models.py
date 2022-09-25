#!/usr/bin/env python

"""
models.py
----------------------------------
Scintillation models

A library of scintillation models to use with lmfit, emcee, or bilby

    Each model has at least inputs:
        params
        xdata
        ydata
        weights

    And output:
        residuals = (ydata - model) * weights

    Some functions use additional inputs
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scintools.scint_sim import ACF
from scipy.ndimage import gaussian_filter


def powerspectrum_model(params, wavenumber):

    amp = params['amp']
    wn = params['wn']
    alpha = params['alpha']

    return wn + amp * wavenumber**alpha


def tau_acf_model(params, time_lag):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        alpha = index of exponential function. 2 is Gaussian, 5/3 is Kolmogorov
        wn = white noise spike in ACF cut
    """

    amp = params['amp']
    tau = params['tau']
    alpha = params['alpha']
    wn = params['wn']

    model = amp*np.exp(-np.divide(time_lag, tau)**(alpha))
    model[0] += wn  # add white noise spike

    # Multiply by triangle function
    return np.multiply(model, 1-np.divide(time_lag, max(time_lag)))


def dnu_acf_model(params, frequency_lag):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        dnu = bandwidth at 1/2 power
        wn = white noise spike in ACF cut
    """

    amp = params['amp']
    dnu = params['dnu']
    wn = params['wn']

    model = amp*np.exp(-np.divide(frequency_lag, dnu/np.log(2)))
    model[0] += wn  # add white noise spike

    # Multiply by triangle function
    return np.multiply(model, 1-np.divide(frequency_lag, max(frequency_lag)))


def scint_acf_model(params, time_lag, frequency_lag):
    """
    Fit both tau (tau_acf_model) and dnu (dnu_acf_model) simultaneously
    """

    model_t = tau_acf_model(params, time_lag)
    model_f = dnu_acf_model(params, frequency_lag)

    return np.concatenate((model_t, model_f))


def scint_acf_model_2d_approx(params, time_lag, frequency_lag):
    """
    Fit an approximate 2D ACF function
    """

    amp = params['amp']
    dnu = params['dnu']
    tau = params['tau']
    alpha = params['alpha']
    mu = params['phasegrad'] * 60  # min/MHz to s/MHz
    tobs = params['tobs']
    bw = params['bw']
    wn = params['wn']

    time_lag = np.reshape(time_lag, (len(time_lag), 1))
    frequency_lag = np.reshape(frequency_lag, (1, len(frequency_lag)))

    model = amp * np.exp(
        -(abs((time_lag - mu*frequency_lag)/tau)**(3 * alpha / 2) +
          abs(frequency_lag / (dnu / np.log(2)))**(3 / 2))**(2 / 3))

    # multiply by triangle function
    model = np.multiply(model, 1-np.divide(abs(time_lag), tobs))
    model = np.multiply(model, 1-np.divide(abs(frequency_lag), bw))
    model = np.fft.fftshift(model)
    model[-1, -1] += wn  # add white noise spike
    model = np.fft.ifftshift(model)

    return np.transpose(model)


def scint_acf_model_2d(params, dim):
    """
    Fit an analytical 2D ACF function
    """

    tau = np.abs(params['tau'])
    dnu = np.abs(params['dnu'])
    alpha = params['alpha']
    ar = np.abs(params['ar'])
    psi = params['psi']
    phasegrad = params['phasegrad']
    theta = params['theta']
    wn = params['wn']
    amp = params['amp']

    tobs = params['tobs']
    bw = params['bw']
    nt = params['nt']
    nf = params['nf']

    nf_crop, nt_crop = dim
    tobs_crop, bw_crop = (nt_crop / nt) * tobs, (nf_crop / nf) * bw

    taumax = tobs_crop / tau
    dnumax = bw_crop / dnu

    acf = ACF(taumax=taumax, dnumax=dnumax, nt=nt_crop, nf=nf_crop, ar=ar,
              alpha=alpha, phasegrad=phasegrad, theta=theta,
              amp=amp, wn=wn, psi=psi)
    acf.calc_acf()
    model = acf.acf

    triangle_t = 1 - np.divide(np.tile(np.abs(np.linspace(
        -taumax*tau, taumax*tau, nt_crop)), (nf_crop, 1)), tobs)
    triangle_f = np.transpose(1 - np.divide(np.tile(np.abs(np.linspace(
        -dnumax*dnu, dnumax*dnu, nf_crop)), (nt_crop, 1)), bw))
    triangle = np.multiply(triangle_t, triangle_f)

    # multiply by triangle function
    return np.multiply(model, triangle)


def tau_sspec_model(params, time_lag):
    """
    Fit 1D function to cut through ACF for scintillation timescale.
    Exponent is 5/3 for Kolmogorov turbulence.
        amp = Amplitude
        tau = timescale at 1/e
        alpha = index of exponential function. 2 is Gaussian, 5/3 is Kolmogorov
        wn = white noise spike in ACF cut
    """

    amp = params['amp']
    tau = params['tau']
    alpha = params['alpha']
    wn = params['wn']

    model = amp * np.exp(-np.divide(time_lag, tau)**alpha)
    model[0] += wn  # add white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1 - np.divide(time_lag, max(time_lag)))

    model_flipped = model[::-1]
    model = np.concatenate((model, model_flipped))
    model = model[0:2 * len(time_lag) - 1]
    # Get Fourier model
    model = np.fft.fft(model)
    model = np.real(model)

    return model[0:len(time_lag)]


def dnu_sspec_model(params, frequency_lag):
    """
    Fit 1D function to cut through ACF for decorrelation bandwidth.
    Default function has is exponential with dnu measured at half power
        amp = Amplitude
        dnu = bandwidth at 1/2 power
        wn = white noise spike in ACF cut
    """

    amp = params['amp']
    dnu = params['dnu']
    wn = params['wn']

    model = amp * np.exp(-np.divide(frequency_lag, dnu / np.log(2)))
    model[0] += wn  # add white noise spike
    # Multiply by triangle function
    model = np.multiply(model, 1-np.divide(frequency_lag, max(frequency_lag)))

    model_flipped = model[::-1]
    model = np.concatenate((model, model_flipped))
    model = model[0:2 * len(frequency_lag) - 1]
    # Get Fourier model
    model = np.fft.fft(model)
    model = np.real(model)

    return model[0:len(frequency_lag)]


def scint_sspec_model(params, time_lag, frequency_lag):
    """
    Fit both tau (tau_acf_model) and dnu (dnu_acf_model) simultaneously
    """

    model_t = tau_sspec_model(params, time_lag)
    model_f = dnu_sspec_model(params, frequency_lag)

    return np.concatenate((model_t, model_f))


def arc_power_curve(params, xdata):
    """
    Returns a template for the power curve in secondary spectrum vs
    sqrt(curvature) or normalised fdop
    """

    model = []

    return model


def fit_parabola(x, y):
    """
    Fit a parabola and return the value and error for the peak
    """

    # increase range to help fitter
    ptp = np.ptp(x)
    x = x*(1000/ptp)

    # Do the fit
    params, pcov = np.polyfit(x, y, 2, cov=True)
    yfit = params[0]*np.power(x, 2) + params[1]*x + params[2]  # y values

    # Get parameter errors
    errors = []
    for i in range(len(params)):  # for each parameter
        errors.append(np.absolute(pcov[i][i])**0.5)

    # Get parabola peak and error
    peak = -params[1]/(2*params[0])  # Parabola max (or min)
    peak_error = np.sqrt((errors[1]**2)*((1/(2*params[0]))**2) +
                         (errors[0]**2)*((params[1]/2)**2))  # Error on peak

    peak = peak*(ptp/1000)
    peak_error = peak_error*(ptp/1000)

    return yfit, peak, peak_error


def fit_log_parabola(x, y):
    """
    Fit a log-parabola and return the value and error for the peak
    """

    # Take the log of x
    logx = np.log(x)
    ptp = np.ptp(logx)
    x = logx*(1000/ptp)  # increase range to help fitter

    # Do the fit
    yfit, peak, peak_error = fit_parabola(x, y)
    frac_error = peak_error/peak

    peak = np.e**(peak*ptp/1000)
    # Average the error asymmetries
    peak_error = frac_error*peak

    return yfit, peak, peak_error


def arc_curvature(params, true_anomaly, vearth_ra, vearth_dec, mjd=None,
                  return_veff=False):
    """
    arc curvature model

    """

    # ensure dimensionality of arrays makes sense
    if true_anomaly.ndim > 1:
        true_anomaly = true_anomaly.squeeze()
        vearth_ra = vearth_ra.squeeze()
        vearth_dec = vearth_dec.squeeze()

    kmpkpc = 3.085677581e16
    dkm = params['d'] * kmpkpc  # pulsar distance in kpc
    s = params['s']

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, true_anomaly,
                                  vearth_ra, vearth_dec, mjd=mjd)

    if 'psi' in params.keys():
        raise KeyError("parameter psi is no longer supported. Please use zeta")
    if 'vism_psi' in params.keys():
        raise KeyError("parameter vism_psi is no longer supported. " +
                       "Please use vism_zeta")

    if 'nmodel' in params.keys():
        nmodel = params['nmodel']
    else:
        if 'zeta' in params.keys():
            nmodel = 1
        else:
            nmodel = 0

    if 'vism_ra' in params.keys():
        vism_ra = params['vism_ra']
        vism_dec = params['vism_dec']
    else:
        vism_ra = 0
        vism_dec = 0

    if nmodel > 0.5:
        zeta = params['zeta'] * np.pi / 180
        if 'vism_zeta' in params.keys():
            vism_zeta = params['vism_zeta']
            veff2 = (veff_ra*np.sin(zeta) + veff_dec*np.cos(zeta) -
                     vism_zeta)**2
        else:
            veff2 = ((veff_ra - vism_ra) * np.sin(zeta) +
                     (veff_dec - vism_dec) * np.cos(zeta)) ** 2
    else:
        veff2 = (veff_ra - vism_ra)**2 + (veff_dec - vism_dec)**2

    model = dkm * s * (1 - s)/(2 * veff2)
    model = model/1e9  # convert to 1/(m * mHz**2)

    if return_veff:
        return model, (veff_ra - vism_ra), (veff_dec - vism_dec)
    else:
        return model


def veff_thin_screen(params, true_anomaly, vearth_ra, vearth_dec, mjd=None):
    """
    Effective velocity thin screen model.
    Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.

    """

    # ensure dimensionality of arrays makes sense
    if true_anomaly.ndim > 1:
        true_anomaly = true_anomaly.squeeze()
        vearth_ra = vearth_ra.squeeze()
        vearth_dec = vearth_dec.squeeze()

    s = params['s']
    d = params['d']
    if 'kappa' in params.keys():
        kappa = params['kappa']
    else:
        kappa = 1

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params, true_anomaly,
                                  vearth_ra, vearth_dec, mjd=mjd)

    if 'nmodel' in params.keys():
        nmodel = params['nmodel']
    else:
        if 'psi' in params.keys():
            nmodel = 1
        else:
            nmodel = 0

    if 'vism_ra' in params.keys():
        vism_ra = params['vism_ra']
        vism_dec = params['vism_dec']
    else:
        vism_ra = 0
        vism_dec = 0

    veff_ra -= vism_ra
    veff_dec -= vism_dec

    if nmodel > 0.5:
        R = params['R']
        psi = params['psi'] * np.pi / 180

        cosa = np.cos(2 * psi)
        sina = np.sin(2 * psi)

        # quadratic coefficients
        a = (1 - R * cosa) / np.sqrt(1 - R**2)
        b = (1 + R * cosa) / np.sqrt(1 - R**2)
        c = -2 * R * sina / np.sqrt(1 - R**2)

    else:
        a, b, c = 1, 1, 0

    # coefficient to match model with data
    coeff = 1 / np.sqrt(2 * d * (1 - s) / s)

    veff = kappa * (np.sqrt(a*veff_dec**2 + b*veff_ra**2 +
                            c*veff_ra*veff_dec))

    return coeff * veff / s


def effective_velocity_annual(params, true_anomaly, vearth_ra, vearth_dec,
                              mjd=None):
    """
    Effective velocity with annual and pulsar terms
        Note: Does NOT include IISM velocity, but returns veff in IISM frame
    """
    # Define some constants
    v_c = 299792.458  # km/s
    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    # tempo2 parameters from par file in capitals
    if 'PB' in params.keys():
        A1 = params['A1']  # projected semi-major axis in lt-s
        PB = params['PB']  # orbital period in days
        ECC = params['ECC']  # orbital eccentricity
        OM = params['OM'] * np.pi/180  # longitude of periastron rad
        if 'OMDOT' in params.keys():
            if mjd is None:
                print('Warning, OMDOT present but no mjd for calculation')
                omega = OM
            else:
                omega = OM + \
                    params['OMDOT']*np.pi/180*(mjd-params['T0'])/365.2425
        else:
            omega = OM
        # Note: fifth Keplerian param T0 used in true anomaly calculation
        if 'KIN' in params.keys():
            INC = params['KIN']*np.pi/180  # inclination
        elif 'COSI' in params.keys():
            INC = np.arccos(params['COSI'])
        elif 'SINI' in params.keys():
            INC = np.arcsin(params['SINI'])
        else:
            print('Warning: inclination parameter (KIN, COSI, or SINI) ' +
                  'not found')

        if 'sense' in params.keys():
            sense = params['sense']
            if sense < 0.5:  # KIN < 90
                if INC > np.pi/2:
                    INC = np.pi - INC
            if sense >= 0.5:  # KIN > 90
                if INC < np.pi/2:
                    INC = np.pi - INC

        KOM = params['KOM']*np.pi/180  # longitude ascending node

        # Calculate pulsar velocity aligned with the line of nodes (Vx) and
        #   perpendicular in the plane (Vy)
        vp_0 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
                                         np.sqrt(1 - ECC**2))
        vp_x = -vp_0 * (ECC * np.sin(omega) + np.sin(true_anomaly + omega))
        vp_y = vp_0 * np.cos(INC) * (ECC * np.cos(omega) + np.cos(true_anomaly
                                                                  + omega))
    else:
        vp_x = 0
        vp_y = 0

    if 'PMRA' in params.keys():
        PMRA = params['PMRA']  # proper motion in RA
        PMDEC = params['PMDEC']  # proper motion in DEC
    else:
        PMRA = 0
        PMDEC = 0

    # other parameters in lower-case
    s = params['s']  # fractional screen distance
    d = params['d']  # pulsar distance in kpc
    d = d * kmpkpc  # distance in km

    pmra_v = PMRA * masrad * d / secperyr
    pmdec_v = PMDEC * masrad * d / secperyr

    # Rotate pulsar velocity into RA/DEC
    vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
    vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

    # find total effective velocity in RA and DEC
    veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v)
    veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v)

    return veff_ra, veff_dec, vp_ra, vp_dec


def arc_weak(ftn, ar=1, psi=0, alpha=11/3):
    """
    Parameters
    ----------
    ftn : Array 1D
        The normalised Doppler frequency (x-axis), where ftn=1 is the arc

    ar : float, optional
        Anisotropy axial ratio. The default is 1.
    psi : float, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    p : Array 1D
        The model poppler profile

    """

    # Begin model
    a = np.cos(psi * np.pi/180)**2 / ar + ar * np.sin(psi*np.pi/180)**2
    b = ar * np.cos(psi * np.pi/180)**2 + (np.sin(psi * np.pi/180)**2)/ar
    c = 2*np.sin(psi * np.pi/180)*np.cos(psi * np.pi/180)*(1/ar - ar)

    p = ((a*ftn**2 + b*(1 - ftn**2) + c*ftn*(1 - ftn**2)**0.5)**(-alpha/2) +
         (a*ftn**2 + b*(1 - ftn**2) - c*ftn*(1 - ftn**2)**0.5)**(-alpha/2))
    p /= np.sqrt(1 - ftn**2)

    return p


def arc_weak_2d(fdop, tdel, eta=1, ar=1, psi=0, alpha=11/3):
    """
    Parameters
    ----------
    fdop : Array 1D
        The Doppler frequency (x-axis) coordinates of the model secondary
        spectrum.
    tdel : Array 1D
        The wavenumber (y-axis) coordinates of the model secondary spectrum.
    eta : floar, optional
        Arc curvature. The default is 1.
    ar : float, optional
        Anisotropy axial ratio. The default is 1.
    psi : float, optional
        DESCRIPTION. The default is 0.
    alpha : float, optional
        DESCRIPTION. The default is 11/3.

    Returns
    -------
    sspec : Array 2D
        The model secondary spectrum.

    """

    # Begin model
    a = np.cos(psi * np.pi/180)**2 / ar + ar * np.sin(psi*np.pi/180)**2
    b = ar * np.cos(psi * np.pi/180)**2 + (np.sin(psi * np.pi/180)**2)/ar
    c = 2*np.sin(psi * np.pi/180)*np.cos(psi * np.pi/180)*(1/ar - ar)

    fdx, TDEL = np.meshgrid(fdop, tdel)

    f_arc = np.sqrt(TDEL/eta)

    fdy = np.sqrt(TDEL/eta - fdx**2)

    p = (a*fdx**2 + b*fdy**2 + c*fdx*fdy)**(-11/6) + \
        (a*fdx**2 + b*fdy**2 - c*fdx*fdy)**(-11/6)

    arc_frac = np.real(fdx)/np.real(f_arc)
    sspec = p / np.sqrt(1 - arc_frac**2)

    return sspec
