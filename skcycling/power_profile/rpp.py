"""Record power-profile

This module contains class and methods related to the record power-profile.
"""
import numpy as np

import os
import warnings

from scipy.interpolate import interp1d

from joblib import Parallel, delayed

from ..utils.checker import _check_X
from ..utils.fit import log_linear_fitting
from ..utils.fit import log_linear_model


SAMPLING_WKO = np.array([0.016, 0.083, 0.5, 1, 3, 3.5, 4, 4.5, 5, 5.5,
                         6, 6.5, 7, 10, 20, 30, 45, 60, 120, 180, 240])


def _rpp_parallel(X, idx_t_rpp):
    """ Function to compute the rpp in parallel

    Parameters
    ----------
    X : ndarray, shape (n_samples)
        The power records.

    idx_t_rpp : int
        Index of the time to compute the rpp.

    Return
    ------
    power : float
        Returns the best power for the given duration of the rpp.
    """
    # Slice the data such that we can compute efficiently the mean later
    t_crop = np.array([X[i:-idx_t_rpp + i:]
                       for i in range(idx_t_rpp)])
    # Check that there is some value cropped. In the case that
    # the duration is longer than the file, the table crop is
    # empty
    if t_crop.size is not 0:
        # Compute the mean for each of these samples
        t_crop_mean = np.mean(t_crop, axis=0)
        # Keep the best to store as rpp
        return np.max(t_crop_mean)
    else:
        return 0


def compute_ride_rpp(X, max_duration_rpp, in_parallel=True):
    """ Compute the record power-profile

    Parameters
    ----------
    X : array-like, shape (n_samples, )

    in_parallel : boolean
        If True, the rpp will be computed on all the available cores.

    Return
    ------
    rpp : array-like, shape (n_samples, )
        Array containing the record power-profile of the current ride.
    """

    # Check that X is proper
    X = _check_X(X)

    if in_parallel is not True:
        # Initialize the ride rpp
        rpp = np.zeros(60 * max_duration_rpp)

        # For each duration in the rpp
        for idx_t_rpp in range(rpp.size):
            # Slice the data such that we can compute efficiently
            # the mean later
            t_crop = np.array([X[i:-idx_t_rpp + i:]
                               for i in range(idx_t_rpp)])
            # Check that there is some value cropped. In the case that
            # the duration is longer than the file, the table crop is
            # empty
            if t_crop.size is not 0:
                # Compute the mean for each of these samples
                t_crop_mean = np.mean(t_crop, axis=0)
                # Keep the best to store as rpp
                rpp[idx_t_rpp] = np.max(t_crop_mean)

        return rpp

    else:
        rpp = Parallel(n_jobs=-1)(delayed(_rpp_parallel)(X, idx_t_rpp)
                                  for idx_t_rpp
                                  in range(60 * max_duration_rpp))
        # We need to make a conversion from list to numpy array
        return np.array(rpp)


class Rpp(object):
    """ Record power-profile

    Can perform online updates via `partial_fit` method.

    Parameters
    ----------
    max_duration_rpp : int
        Integer representing the maximum duration in minutes to
        build the record power-profile model.

    cyclist_weight : float, default None
        Float in order to normalise the record power-profile depending
        of its weight. By default this is None in order to avoid
        using the data from normalized rpp without this data.

    Attributes
    ----------
    rpp_ : array-like, shape (60 * max_duration_rpp, )
        Array in which the record power-profile is stored.
        The units used is the second.

    rpp_norm_ : array-like, shape (60 * max_duration_rpp, )
        Array in which the weight-normalized record power-profile
        is stored. The units used is the seconds.

    max_duration_rpp_ : int
        The maximum duration of the record power-profile.

    cyclist_weight_ : float
        Cyclist weight.
    """

    def __init__(self, max_duration_rpp=None, cyclist_weight=None):
        self.max_duration_rpp = max_duration_rpp
        self.cyclist_weight = cyclist_weight

    def _check_partial_fit_first_call(self):
        """ Private function helper to know if the record power-profile
            already has been computed.
        """

        # Check if the record power-profile was previously created
        if getattr(self, 'rpp_', None) is not None:
            # Not the first time that the fitting is called
            return False
        else:
            # First time that the fitting is called
            # Initalise the rpp_ variable
            if self.max_duration_rpp is None:
                raise ValueError('You should instantiate the object with a'
                                 ' max duration for the rpp.')
            self.max_duration_rpp_ = self.max_duration_rpp
            self.cyclist_weight_ = self.cyclist_weight
            self.rpp_ = np.zeros(60 * self.max_duration_rpp)
            # If the weight is not None we can also initialize the
            # normalized rpp
            if self.cyclist_weight_ is not None:
                self.rpp_norm_ = self.rpp_.copy()
            else:
                self.rpp_norm_ = None

            return True

    def load_from_npy(self, filename, cyclist_weight=None):
        """ Load the record power-profile from an npy file

        Parameters
        ----------
        filename : str
            String containing the path to the NPY file containing the array
            representing the record power-profile.

        cyclist_weight : float or None, default None
            Float in order to normalise the record power-profile depending
            of its weight. By default this is None in order to avoid
            using the data from normalized rpp without this data.

        Return
        ------
        self : object
            Returns self
        """

        # Check that the file exist first
        if not os.path.isfile(filename):
            raise ValueError('The file does not exist.')

        # Check that the file is an npy file
        if not filename.endswith('.npy'):
            raise ValueError('The file should be an npy file.')

        # Check that the cyclist weight is a float
        if not (cyclist_weight is None or isinstance(cyclist_weight, float)):
            raise ValueError('The cyclist weight need to be a float.')

        # Load the record power-profile
        self.rpp_ = np.load(filename)

        # We have to infer the duration of the rpp
        max_duration_rpp = self.rpp_.size / 60
        self.max_duration_rpp_ = max_duration_rpp

        # Apply the cyclist weight
        self.cyclist_weight_ = cyclist_weight

        # Compute the normalized rpp if possible
        if self.cyclist_weight_ is not None:
            self.rpp_norm_ = self.rpp_ / self.cyclist_weight_
        else:
            self.rpp_norm_ = None

        return self

    def fit(self, X, in_parallel=True):
        """ Fit the data to the RPP

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores.

        Return
        ------
        self : object
            Returns self.

        """

        # We should check if X is proper
        X = _check_X(X)

        # Make a partial fitting of the current data
        return self.partial_fit(X, refit=False, in_parallel=in_parallel)

    def partial_fit(self, X, refit=False, in_parallel=True):
        """ Incremental fit of the RPPB

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores.

        Return
        ------
        self : object
            Returns self.

        """

        # Check that X is proper
        X = _check_X(X)

        # Call the partial fitting
        return self._partial_fit(X, refit=refit, in_parallel=in_parallel)

    def _partial_fit(self, X, refit=False, in_parallel=True):
        """ Actual implementation of RPP calculation

        Parameters
        ----------
        X : array-like, shape (n_samples, )

        refit : bool
            If True, the RPP will be overidden.

        in_parallel : boolean
            If True, the rpp will be computed on all the available cores.

        Return
        ------
        self : object
            Returns self.

        """

        # We should check X
        X = _check_X(X)

        # If we want to recompute the record power-profile
        if refit:
            self.rpp_ = None
            self.rpp_norm_ = None

        # Check if this the first called or if we have to initialize
        # the record power-profile
        if self._check_partial_fit_first_call():
            # What to do if it was the first call
            # Compute the record power-profile for the given X
            self.rpp_ = compute_ride_rpp(X,
                                         self.max_duration_rpp_,
                                         in_parallel)
            # Compute the normalized rpp if we should
            if self.cyclist_weight_ is not None:
                self.rpp_norm_ = self.rpp_ / self.cyclist_weight_
        else:
            # What to do if it was yet another call
            # Compute the record power-profile for the given X
            self.rpp = compute_ride_rpp(X,
                                        self.max_duration_rpp_,
                                        in_parallel)
            # Update the best record power-profile
            self.rpp_ = np.max((self.rpp, self.rpp_), axis=0)
            # Compute the normalized rpp if we should
            if self.cyclist_weight_ is not None:
                self.rpp_norm_ = self.rpp_ / self.cyclist_weight_
            else:
                self.rpp_norm_ = None

        return self

    # def denoise_rpp(self, method='b-spline', normalized=False):
    #     """ Denoise the record power-profile

    #     Parameters
    #     ----------
    #     method : str, default 'b-spline'
    #         Method to select to denoise the record power-profile.

    #     normalized : bool, default False
    #         Return a weight-normalized rpp if True.

    #     Return
    #     ------
    #     rpp : array-like, shape (n_samples, )
    #         Return a denoise record power-profile.
    #     """

    #     if method == 'b-spline':
    #         # Shall used the rpp or weight-normalized rpp
    #         if normalized is True:
    #             # Check that the cyclist weight was provided
    #             if self.cyclist_weight_ is not None:
    #                 rpp = self.rpp_norm_
    #             else:
    #                 raise ValueError('You cannot get a normalized rpp if the'
    #                                  ' cyclist weight never has been given.')
    #         else:
    #             rpp = self.rpp_

    #         # Apply denoising based on b-spline
    #         # Create the timeline
    #         t = np.linspace(0, self.max_duration_rpp_, rpp.size)
    #         spl = UnivariateSpline(t, rpp)

    #         return spl(t)
    #     else:
    #         raise ValueError('This denoising method is not implemented.')

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
        rpp : array-like, shape (n_samples, )
            Return a resampled record power-profile.
        """

        # Shall used the rpp or weight-normalized rpp
        if normalized is True:
            # Check that the cyclist weight was provided
            if self.cyclist_weight_ is not None:
                rpp = self.rpp_norm_
            else:
                raise ValueError('You cannot get a normalized rpp if the'
                                 ' cyclist weight never has been given.')
        else:
            rpp = self.rpp_

        t = np.linspace(0, self.max_duration_rpp_, rpp.size)
        f = interp1d(t, rpp, kind=method_interp)

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

        if np.count_nonzero(ts > self.max_duration_rpp_) > 0:
            # The values which are outside of the maximum duration need to
            # be removed
            ts = ts[np.nonzero(ts <= self.max_duration_rpp_)]
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
