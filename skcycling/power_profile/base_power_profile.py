""" Basic class for power profile.
"""

import warnings
import numpy as np
import cPickle as pickle

from scipy.interpolate import interp1d

from abc import ABCMeta, abstractmethod

from ..utils.fit import log_linear_fitting
from ..utils.fit import log_linear_model

SAMPLING_WKO = np.array([0.016, 0.083, 0.5, 1, 3, 3.5, 4, 4.5, 5, 5.5,
                         6, 6.5, 7, 10, 20, 30, 45, 60, 120, 180, 240])


class BasePowerProfile(object):
    """ Basic class for power profile.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, max_duration_profile=None, cyclist_weight=None):
        """ Constructor. """
        self.max_duration_profile_ = max_duration_profile
        self.cyclist_weight_ = cyclist_weight

    def load_from_pickles(self, filename):
        """ Function to load an object RecordPowerProfile through pickles.

        Parameters
        ----------
        filename : str
            Filename to the pickle file.

        Return
        ------
        self : object
            Returns self.
        """
        self = pickle.load(open(filename, 'rb'))

        return self

    def save_to_pickles(self, filename):
        """ Function to save an object RecordPowerProfile through pickles.

        Parameters
        ----------
        filename : str
            Filename to the pickle file.

        Return
        ------
        None
        """
        pickle.dump(self, open(filename, 'wb'))

        return None

    @abstractmethod
    def fit(self):
        """ Method to compute the power profile. """
        raise NotImplementedError

    def resampling_rpp(self, ts, method_interp='linear', normalized=False):
        """ Resampling the record power-profile

        Parameters
        ----------
        ts : array-like, shape (n_sample, )
            An array containaining the time landmark to sample.

        method_interp : string, default 'linear'
            Name of the method to interpolate the data.
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
            where 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of first, second or third order) or as an integer
            specifying the order of the spline interpolator to use.

        normalized : bool, default False
            Return a weight-normalized rpp if True.

        Return
        ------
        data : array-like, shape (n_samples, )
            Return a resampled record power-profile.
        """

        # Shall used the rpp or weight-normalized rpp
        if normalized is True:
            # Check that the cyclist weight was provided
            if self.cyclist_weight_ is not None:
                data = self.data_norm_
            else:
                raise ValueError('You cannot get a normalized rpp if the'
                                 ' cyclist weight never has been given.')
        else:
            data = self.data_

        t = np.linspace(0, self.max_duration_profile_, data.size)
        f = interp1d(t, data, kind=method_interp)

        return f(ts)

    def aerobic_meta_model(self, ts=None, normalized=False, method='lsq'):
        """ Compute the aerobic metabolism model from the
            record power-profile
        Parameters
        ----------
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

        # If ts is not provided we have to create a timeline
        if ts is None:
            # By default ts will be taken as in WKO+
            ts = SAMPLING_WKO.copy()

        if np.count_nonzero(ts > self.max_duration_profile_) > 0:
            # The values which are outside of the maximum duration need to
            # be removed
            ts = ts[np.nonzero(ts <= self.max_duration_profile_)]
            warnings.warn('Samples in `ts` have been removed since that they'
                          ' are not information inside the rpp.')

        # Compute the rpp
        rpp = self.resampling_rpp(ts, normalized=normalized)

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
