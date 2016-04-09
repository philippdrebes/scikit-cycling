""" Rider class.

This module contains the class to manage the data linked to a single rider.
"""

import os
import warnings
import numpy as np
import cPickle as pickle

from datetime import date

from ..power_profile import RidePowerProfile
from ..power_profile import RecordPowerProfile

from ..utils.checker import check_filename_pickle_load
from ..utils.checker import check_filename_pickle_save
from ..utils.checker import check_filename_fit
from ..utils.checker import check_tuple_date


class Rider(object):
    """ Rider class to aggregate all the different power tools.

    Parameters
    ----------
    max_duration_profile : int
        Integer representing the maximum duration in minutes to build the
        record power-profile model. It can be infered if the data are loaded
        from a pickle file.

    cyclist_weight : float
        Float in order to normalise the record power-profile depending
        of its weight.

    rides_pp : list of RidePowerProfile or None
        Initialize the list of RidePowerProfile

    Attributes
    ----------
    rides_pp_ : list of RidePowerProfile
        The list of the ride power-profile linked to a rider.

    record_pp_ : RecordPowerProfile
        The record power-profile of the rider.

    max_duration_profile_ : int
        Integer representing the maximum duration in minutes to build the
        record power-profile model.

    cyclist_weight_ : float
        Float in order to normalise the record power-profile depending
        of its weight.
    """

    def __init__(self, cyclist_weight, max_duration_profile, rides_pp=None):
        """ Constructor. """
        self.cyclist_weight_ = cyclist_weight
        self.max_duration_profile_ = max_duration_profile
        # Check the list of ride
        if rides_pp is None:
            # Initialize the list to an empty list
            self.rides_pp_ = []
        else:
            # Check if all the elements are from the class RidePowerProfile
            self.rides_pp_ = self._validate_rides_pp(rides_pp)

        # Initialize the record_pp by default
        self.record_pp_ = RecordPowerProfile(max_duration_profile=self.max_duration_profile_,
                                             cyclist_weight=self.cyclist_weight_)

    def _validate_rides_pp(self, rides_pp):
        """ Method to check the consistency of the ride power-profile list.

        Parameters
        ----------
        rides_pp : list of RidePowerProfile
            Normally a list of RidePowerProfile.

        Return
        ------
        rides_pp : list of RidePowerProfile
            Return the validated list of RidePowerProfile.
        """
        # Check that this is a list
        if isinstance(rides_pp, list):
            # Check that each element are from the class RidePowerProfile
            for rpp in rides_pp:
                if not isinstance(rpp, RidePowerProfile):
                    raise ValueError('The object in the list need to be from'
                                     ' the type RidePowerProfile')
                # We need to check that each ride has been fitted
                if getattr(rpp, 'data_', None) is None or rpp.max_duration_profile_ is None:
                    raise ValueError('One of the ride never has been fitted.'
                                     ' Fit before to compute the record rpp.')
            # Create a list of all the max duration to check that they are
            # all equal
            max_duration = np.array([rpp.max_duration_profile_
                                     for rpp in rides_pp])
            if not np.all(max_duration == self.max_duration_profile_):
                raise ValueError('The maximum duration of the profile should'
                                 ' be the same for all the data.')
            return rides_pp
        else:
            raise ValueError('The ride power-profile should be given as'
                             ' a list.')

    @staticmethod
    def load_from_pickles(filename):
        """ Function to load an object through pickles.

        Parameters
        ----------
        filename : str
            Filename to the pickle file. The extension should be `.p`.

        Return
        ------
        bpp : object
            Returns Rider.
        """
        # Check the consistency of the filename
        filename = check_filename_pickle_load(filename)
        # Load the pickle
        bpp = pickle.load(open(filename, 'rb'))

        return bpp

    def save_to_pickles(self, filename):
        """ Function to save an object RecordPowerProfile through pickles.

        Parameters
        ----------
        filename : str
            Filename to the pickle file. The extension should be `.p`.

        Return
        ------
        None
        """
        # We need to check that the directory where the file will be exist
        dir_pickle = os.path.dirname(filename)
        if not os.path.exists(dir_pickle):
            os.makedirs(dir_pickle)
        # Check the consistency of the filename
        filename = check_filename_pickle_save(filename)
        # Create the pickle file
        pickle.dump(self, open(filename, 'wb'))

        return None

    def add_rides_from_path(self, path, overwrite=False, verbose=True):
        """ Function which allows to read and fit some ride and add them to
        the current list of ride.

        Parameters
        ----------
        path : str
            Path from where to get the fit file.

        overwrite : bool
            Overwrite the current ride power-profile list.

        verbose : bool
            Show information of the process.

        Return:
        self : object
            Returns self.
        """
        # Check that the path exist
        if not os.path.exists(path):
            raise ValueError('The path is not existing.')

        # Find all the fit file
        rides_rpp = []
        for filename in os.listdir(path):
            # Take only the fit file
            if filename.endswith('.fit'):
                if verbose:
                    print 'Process the file: {}'.format(filename)
                rpp = RidePowerProfile(max_duration_profile=self.max_duration_profile_,
                                       cyclist_weight=self.cyclist_weight_)
                rpp.fit(os.path.join(path, filename))
                rides_rpp.append(rpp)

        # Check if we have to overwrite the list
        if overwrite:
            self.rides_pp_ = rides_rpp
        else:
            self.rides_pp_ += rides_rpp

        return self

    def add_ride_from_fit(self, filename):
        """ Function to add one ride to the list using fit file.

        Parameters
        ----------
        filename : str
            Filename of the fit file to add.

        Return
        ------
        self : obj
            Returns self.
        """
        # Check that the filename is a fit file and is consitant
        filename = check_filename_fit(filename)

        # Create an object to handle the ride
        rpp = RidePowerProfile(max_duration_profile=self.max_duration_profile_,
                                       cyclist_weight=self.cyclist_weight_)
        rpp.fit(filename)

        # Add to the current file
        self.rides_pp_.append(rpp)

        return self

    def delete_ride(self, date_ride):
        """ Function to delete a specific ride from the list.

        Parameters
        ----------
        date_ride : date
            The date of the ride to remove.

        Return
        ------
        self : object
            Returns self.
        """
        # Check the consistency of the date
        if not isinstance(date_ride, date):
            raise ValueError('The date should be a date object.')

        # From the list of ride, get the date of the ride
        date_rides = np.array([rpp.date_profile_ for rpp in self.rides_pp_])

        # Find if there is any date corresponding to the one specified by
        # the user
        if np.count_nonzero(date_rides == date_ride) == 0:
            warnings.warn('No rides have been removed. No matching dates.')
        else:
            idx_rides_keep = np.flatnonzero(date_rides != date_ride)
            self.rides_pp_ = [self.rides_pp_[i] for i in idx_rides_keep]

        return self

    def compute_record_pp(self, date_start_finish=None):
        """ Function to compute the record power profile.

        Parameters
        ----------
        date_start_finish : tuple of date, shape (start, finish)
            Starting and finishing date to consider to compute the date.
            If None, the full range will be considered.

        Return
        ------
        self : object
            Returns self.
        """
        # Check the consistency of the date tuple
        if date_start_finish is not None:
            date_start_finish = check_tuple_date(date_start_finish)

        self.record_pp_.fit(self.rides_pp_, date_profile=date_start_finish)

        return self
