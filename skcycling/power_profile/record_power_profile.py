""" Record Power-Profile class.
"""

import numpy as np

from datetime import date

from .base_power_profile import BasePowerProfile
from .ride_power_profile import RidePowerProfile


class RecordPowerProfile(BasePowerProfile):
    """ Record power-profile

    Parameters
    ----------
    max_duration_profile : int or None
        Integer representing the maximum duration in minutes to build the
        record power-profile model. It can be infered if the data are loaded
        from a npy file.

    cyclist_weight : float, default None
        Float in order to normalise the record power-profile depending
        of its weight. By default this is None in order to avoid
        using the data from normalized rpp without this data.

    Attributes
    ----------
    data_ : array-like, shape (60 * max_duration_profile_, )
        Array in which the record power-profile is stored.
        The units used is the second.

    data_norm_ : array-like, shape (60 * max_duration_rpp, )
        Array in which the weight-normalized record power-profile
        is stored. The units used is the seconds.

    max_duration_data_ : int
        The maximum duration of the record power-profile.

    cyclist_weight_ : float
        Cyclist weight.

    date_profile_ : tuple of date, shape (start, finish)
        Starting and finishing time to compute the record power-profile.
    """

    def __init__(self, max_duration_profile=None,
                 cyclist_weight=None):
        # Call the constructor of the parent class
        super(RecordPowerProfile, self).__init__(max_duration_profile,
                                                 cyclist_weight)

    def _validate_ride_pp(self, ride_pp):
        """ Method to check the consistency of the ride power-profile list.

        Parameters
        ----------
        ride_pp : list of RidePowerProfile
            Normally a list of RidePowerProfile.

        Return
        ------
        ride_pp : list of RidePowerProfile
            Return the validated list of RidePowerProfile.
        """
        # Check that this is a list
        if isinstance(ride_pp, list):
            # Check that each element are from the class RidePowerProfile
            for rpp in ride_pp:
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
                                     for rpp in ride_pp])
            if self.max_duration_profile_ is None:
                raise ValueError('You need to specify the maximum duration for'
                                 ' the profile equal to the maximum duration'
                                 ' of each ride.')
            if not np.all(max_duration == self.max_duration_profile_):
                raise ValueError('The maximum duration of the profile should'
                                 ' be the same for all the data.')
            return ride_pp
        else:
            raise ValueError('The ride power-profile should be given as'
                             ' a list.')

    def _maximal_mean_power(self, ride_pp):
        """ Function to compute the maximal mean power for different time.

        Parameters
        ----------
        ride_pp : list of RidePowerProfile
            The list from the power-profile to consider.

        Return
        ------
        mmp : ndarray, shape (max_duration_profile_, )
            The Maximal Mean Power for the different time
        """
        # From the list of RidePowerProfile, we need to extract the power
        # information
        profile = np.array([pp.data_ for pp in ride_pp])

        return np.max(profile, axis=0)

    def fit(self, ride_pp, date_profile=None):
        """ Function to build the record power-profile from a list of ride
        power-profile.

        Parameters
        ----------
        ride_rpp: list of RidePowerProfile
            The list from which we will compute the record power-profile.

        date_profile: tuple of date, (start, finish)
            The starting and finishing date for which we have to compute the
            record power-profile.

        Return
        ------
        self : object
            Returns self.
        """
        # Check that the ride power-profile list is ok
        ride_pp = self._validate_ride_pp(ride_pp)

        # Check that the date provided are correct
        if date_profile is not None:
            if isinstance(date_profile, tuple) and len(date_profile) == 2:
                # Check that the tuple is of write type
                if isinstance(date_profile[0],
                              date) and isinstance(date_profile[1],
                                                   date):
                    # Check that the first date is earlier than the second date
                    if date_profile[0] < date_profile[1]:
                        date_profile = date_profile
                    else:
                        raise ValueError('The tuple need to be ordered'
                                         ' as (start, finish).')
                else:
                    raise ValueError('Use the class `date` inside the tuple.')
            else:
                raise ValueError('The date are ordered a tuple of'
                                 ' date (start, finsih).')

        # In the case that we want to compute the record power-profile from a
        # subset using the date, we need find the ride which are interesting
        if date_profile is not None:
            # Build a list of the date for each ride
            date_list = np.array([rpp.date_profile_ for rpp in ride_pp])

            # Find the index which are inside the range of date
            idx_ride = np.flatnonzero(np.bitwise_and(
                date_list >= date_profile[0],
                date_list <= date_profile[1]))

            # Compute the record for these files only
            self.data_ = self._maximal_mean_power([ride_pp[i]
                                                   for i in idx_ride])

            # Store the date range
            self.date_profile_ = date_profile
        else:
            # Compute the record for all the files
            self.data_ = self._maximal_mean_power(ride_pp)

            # Store the date range
            date_list = np.array([rpp.date_profile_ for rpp in ride_pp])
            self.date_profile_ = (np.ndarray.min(date_list),
                                  np.ndarray.max(date_list))

        # Compute the normalized rpp if we should
        if self.cyclist_weight_ is not None:
            self.data_norm_ = self.data_ / self.cyclist_weight_
        else:
            self.data_norm_ = None

        return self
