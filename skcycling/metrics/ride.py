""" Metrics to asses the performance of a cycling ride

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

import warnings
import numpy as np

from ..power_profile import BasePowerProfile

from ..restoration.denoise import moving_average

from ..utils.checker import check_X
from ..utils.checker import check_float

from ..utils.fit import log_linear_fitting
from ..utils.fit import log_linear_model

SAMPLING_WKO = np.array([0.016, 0.083, 0.5, 1, 3, 3.5, 4, 4.5, 5, 5.5,
                         6, 6.5, 7, 10, 20, 30, 45, 60, 120, 180, 240])


TS_SCALE_GRAPPE = dict([('I1', 2.), ('I2', 2.5), ('I3', 3.),
                        ('I4', 3.5), ('I5', 4.5), ('I6', 7.),
                        ('I7', 11.)])

ESIE_SCALE_GRAPPE = dict([('I1', (.3, .5)), ('I2', (.5, .6)),
                          ('I3', (.6, .75)), ('I4', (.75, .85)),
                          ('I5', (.85, 1.)), ('I6', (1., 1.80)),
                          ('I7', (1.8, 3.))])


def normalized_power_score(X, pma):
    """ Compute the normalized power for a given ride

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    pma : float
        Maxixum Anaerobic Power.

    Returns
    -------
    score : float
        Return the normalized power.
    """

    # Check the conformity of X and pma
    X = check_X(X)
    pma = check_float(pma)

    # Denoise the rpp through moving average using 30 sec filter
    x_avg = moving_average(X, win=30)

    # Removing value < I1-ESIE, i.e. 30 % PMA
    x_avg = np.delete(x_avg, np.nonzero(x_avg <
                                        (ESIE_SCALE_GRAPPE['I1'][0] * pma)))

    # Compute the mean of the denoised ride elevated
    # at the power of 4
    return np.mean(x_avg ** 4.) ** (1. / 4.)


def intensity_factor_ftp_score(X, ftp):
    """ Compute the intensity factor using the FTP

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    ftp : float
        Functional Threshold Power.

    Returns
    -------
    score: float
        Return the intensity factor.
    """

    # Check the conformity of X and ftp
    X = check_X(X)
    ftp = check_float(ftp)

    # Compute the normalized power
    np_score = normalized_power_score(X, ftp2pma(ftp))

    return np_score / ftp


def intensity_factor_pma_score(X, pma):
    """ Compute the intensity factor using the PMAB

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    pma : float
        Maximum Anaerobic Power.

    Returns
    -------
    score: float
        Return the intensity factor.
    """

    # Check the conformity of X and pma
    X = check_X(X)
    pma = check_float(pma)

    # Compute the resulting IF
    return intensity_factor_ftp_score(X, pma2ftp(pma))


def training_stress_ftp_score(X, ftp):
    """ Compute the training stress score using the FTP

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    ftp : float
        Functional Threshold Power.

    Returns
    -------
    score: float
        Return the training stress score.

    """

    # Check the conformity of X and ftp
    X = check_X(X)
    ftp = check_float(ftp)

    # Compute the intensity factor score
    if_score = intensity_factor_ftp_score(X, ftp)

    # Compute the training stress score
    return (X.size * if_score ** 2) / 3600.


def training_stress_pma_score(X, pma):
    """ Compute the training stress score

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    pma : float
        Maximum Anaerobic Power.

    Returns
    -------
    score: float
        Return the training stress score.
    """

    # Check the conformity of X and pma
    X = check_X(X)
    pma = check_float(pma)

    # Compute the training stress score
    return training_stress_ftp_score(X, pma2ftp(pma))


def pma2ftp(pma):
    """ Convert the PMA to FTP

    Parameter:
    ----------
    pma : float
        Maximum Anaerobic Power.

    Return:
    -------
    ftp : float
        Functioning Threhold Power.
    """

    return 0.76 * pma


def ftp2pma(ftp):
    """ Convert the PMA to FTP

    Parameter:
    ----------
    ftp : float
        Functioning Threhold Power.

    Return:
    -------
    pma : float
        Maximum Anaerobic Power.
    """

    return ftp / 0.76


def training_stress_pma_grappe_score(X, pma):
    """ Compute the training stress score using the PMA,
        considering Grappe et al. approach

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    pma : float
        Maximum Anaerobic Power.

    Returns
    -------
    tss_score: float
        Return the training stress score.
    """

    # Check the consistency of X and pma
    X = check_X(X)
    pma = check_float(pma)

    # Compute the stress for each item of the ESIE
    tss_grappe = 0.
    for key_sc in TS_SCALE_GRAPPE.iterkeys():

        # Count the number of elements which corresponds to as sec
        # We need to convert it to minutes
        curr_stress = np.count_nonzero(
            np.bitwise_and(X >= ESIE_SCALE_GRAPPE[key_sc][0] * pma,
                           X < ESIE_SCALE_GRAPPE[key_sc][1] * pma)) / 60.

        # Compute the cumulative stress
        tss_grappe += curr_stress * TS_SCALE_GRAPPE[key_sc]

    return tss_grappe


def training_stress_ftp_grappe_score(X, ftp):
    """ Compute the training stress score using the FTP,
        considering Grappe et al. approach

    Parameters
    ----------
    X : array-like, shape (n_samples, )
        Array containing the power intensities for a ride.

    ftp : float
        Functional Threshold Power.

    Returns
    -------
    score: float
        Return the training stress score.

    """

    return training_stress_pma_grappe_score(X, ftp2pma(ftp))


def aerobic_meta_model(profile, ts=None, normalized=False, method='lsq'):
    """ Compute the aerobic metabolism model from the
        record power-profile

    Parameters
    ----------
    profile : RidePowerProfile or RecordPowerProfile
        The profile that we want to use for computing the different
        statistics.

    ts : ndarray, shape (n_samples, ) or None
        Array containing the sample to take into account.
        If None, the sampling of the method of Pinot is applied, which is
        equivalent to the sampling from WKO+

    normalized : bool, default False
        Return a weight-normalized rpp if True.

    method : string, default 'lsq'
        Which type of tehcnic to use to make the fitting ('lsq', 'lm').

    Return
    ------
    pma : float
        Maximum Aerobic Power.

    t_pma : int
        Time of the Maximum Aerobic Power in seconds.

    aei : float
        Aerobic Endurance Index.
    Notes
    -----
    [1] Pinot et al., "Determination of Maximal Aerobic Power
    on the Field in Cycling" (2014)
    """
    # Check that the profile is inherating from the class BasePowerProfile
    if not issubclass(type(profile), BasePowerProfile):
        raise ValueError('The variable profile need to be of type'
                         ' RecordPowerProfile or RidePowerProfile.')

    # If ts is not provided we have to create a timeline
    if ts is None:
        # By default ts will be taken as in WKO+
        ts = SAMPLING_WKO.copy()

    if np.count_nonzero(ts > profile.max_duration_profile_) > 0:
        # The values which are outside of the maximum duration need to
        # be removed
        ts = ts[np.nonzero(ts <= profile.max_duration_profile_)]
        warnings.warn('Samples in `ts` have been removed since that they'
                      ' are not information inside the rpp.')

    # Compute the rpp
    rpp = profile.resampling_rpp(ts, normalized=normalized)

    # The zero values need to be avoided for the fitting
    # Keep the signal which is not zero
    ts = ts[np.nonzero(rpp)]
    rpp = rpp[np.nonzero(rpp)]

    # Find the MAP and the corresponding time
    # Only the time between 10 and 240 minutes is used for the regression
    ts_pma_reg = ts[np.nonzero(np.bitwise_and(ts >= 10., ts <= 240.))]
    rpp_pma_reg = rpp[np.nonzero(np.bitwise_and(ts >= 10., ts <= 240.))]

    # Apply the first log-linear fitting
    slope, intercept, std_err, _ = log_linear_fitting(ts_pma_reg,
                                                      rpp_pma_reg,
                                                      method)

    # Find t_pma and pma
    # First record between 3 and 7 min in the confidence area
    ts_pma = ts[np.nonzero(np.bitwise_and(ts >= 3.,
                                          ts <= 7.))]
    rpp_pma = rpp[np.nonzero(np.bitwise_and(ts >= 3.,
                                            ts <= 7.))]

    # Compute the aerobic model found from the regression for
    # the range of interest
    aerobic_model = log_linear_model(ts_pma, slope, intercept)

    # Check the first value which entered in the confidence of 2 std
    if np.count_nonzero(np.abs(rpp_pma -
                               aerobic_model) < 2. * std_err) > 0:
        # Get the first value
        t_pma = ts_pma[np.flatnonzero(np.abs(rpp_pma -
                                             aerobic_model) < 2. *
                                      std_err)[0]]
        # Obtain the corresponding mpa
        pma = rpp_pma[np.flatnonzero(np.abs(rpp_pma -
                                            aerobic_model) < 2. *
                                     std_err)[0]]
    else:
        raise ValueError('There is no value entering in the confidence'
                         ' level between 3 and 7 minutes.')

    # Find aei
    # Get the rpp and ts between t_pma and 240 minutes
    ts_aei_reg = ts[np.nonzero(np.bitwise_and(ts >= float(t_pma),
                                              ts <= 240.))]
    rpp_aei_reg = rpp[np.nonzero(np.bitwise_and(ts >= float(t_pma),
                                                ts <= 240.))]
    # Express the rpp in term of percentage of PMA
    rpp_aei_reg = rpp_aei_reg / float(pma) * 100.

    # Apply a new regression with the aei value
    aei, _, _, _ = log_linear_fitting(ts_aei_reg,
                                      rpp_aei_reg,
                                      method)

    return pma, t_pma, aei
