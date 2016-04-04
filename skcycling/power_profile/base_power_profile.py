""" Basic class for power profile.
"""

import os
import numpy as np
import cPickle as pickle

from scipy.interpolate import interp1d

from abc import ABCMeta, abstractmethod


class BasePowerProfile(object):
    """ Basic class for power profile.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    __metaclass__ = ABCMeta


    @abstractmethod
    def __init__(self, max_duration_profile=None, cyclist_weight=None):
        """ Constructor. """
        self.max_duration_profile = max_duration_profile
        self.cyclist_weight = cyclist_weight

    # @abstractmethod
    # def load_from_npy(self, filename, cyclist_weight=None):
    #     """ Load the record power-profile from an npy file

    #     Parameters
    #     ----------
    #     filename : str
    #         String containing the path to the NPY file containing the array
    #         representing the record power-profile.

    #     cyclist_weight : float or None, default None
    #         Float in order to normalise the record power-profile depending
    #         of its weight. By default this is None in order to avoid
    #         using the data from normalized rpp without this data.

    #     Return
    #     ------
    #     self : object
    #         Returns self
    #     """

    #     # Check that the file exist first
    #     if not os.path.isfile(filename):
    #         raise ValueError('The file does not exist.')

    #     # Check that the file is an npy file
    #     if not filename.endswith('.npy'):
    #         raise ValueError('The file should be an npy file.')

    #     # Check that the cyclist weight is a float
    #     if not (cyclist_weight is None or isinstance(cyclist_weight, float)):
    #         raise ValueError('The cyclist weight need to be a float.')

    #     # Load the record power-profile
    #     self.data_ = np.load(filename)

    #     # We have to infer the duration of the rpp
    #     max_duration_profile = self.data_.size / 60
    #     self.max_duration_profile_ = max_duration_profile

    #     # Apply the cyclist weight
    #     self.cyclist_weight_ = cyclist_weight

    #     # Compute the normalized rpp if possible
    #     if self.cyclist_weight_ is not None:
    #         self.data_norm_ = self.data_ / self.cyclist_weight_
    #     else:
    #         self.data_norm_ = None

    #     return self

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
