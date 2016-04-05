""" Test the rider class. """

import os
import numpy as np

from numpy.testing import assert_equal
from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_warns

from datetime import date

from skcycling.data_management import Rider
from skcycling.power_profile import RidePowerProfile


def test_rider_init_empty():
    """ Test to check the initialisation of rider with empty initilization. """
    rider = Rider(cyclist_weight=60., max_duration_profile=1)

    # Check some attributes
    assert_equal(rider.cyclist_weight_, 60.)
    assert_equal(rider.max_duration_profile_, 1)
    assert_equal(rider.rides_pp_, [])


def test_rider_init_rides():
    """ Test the routine with initilization of a list. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-07-14-26-22.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-11-11-39-38.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-07-26-18-50-56.fit')]

    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=1)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Initialize the rider
    rider = Rider(cyclist_weight=60., max_duration_profile=1,
                  rides_pp=ride_pp_list)

    # Check some attributes
    assert_equal(rider.cyclist_weight_, 60.)
    assert_equal(rider.max_duration_profile_, 1)
    assert_equal(rider.rides_pp_, ride_pp_list)


def test_rider_init_rides_wrong_type():
    """ Test either if an error is raised when the rides power profile is not a
    list. """
    # Initialize the rider
    assert_raises(ValueError, Rider, cyclist_weight=60., max_duration_profile=1,
                  rides_pp=1)


def test_rider_init_rides_wrong_type_in_list():
    """ Test either if an error is raised when the rides power profile is not
    a list of only RidePowerProfile. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-07-14-26-22.fit')]
    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=1)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    ride_pp_list.append(1)

    # Initialize the rider
    assert_raises(ValueError, Rider, cyclist_weight=60., max_duration_profile=1,
                  rides_pp=ride_pp_list)


def test_rider_init_rides_not_init():
    """ Test either if an error is raised when the rides power profile not
    have been initialized. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-07-14-26-22.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-11-11-39-38.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-07-26-18-50-56.fit')]

    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=1)
                    for i in range(len(filename_list))]

    # Initialize the rider
    assert_raises(ValueError, Rider, cyclist_weight=60., max_duration_profile=1,
                  rides_pp=ride_pp_list)


def test_rider_init_rides_wrong_duration():
    """ Test either if an error is raised when the duration are
    not consistent. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-07-14-26-22.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-11-11-39-38.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-07-26-18-50-56.fit')]

    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=2)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Initialize the rider
    assert_raises(ValueError, Rider, cyclist_weight=60., max_duration_profile=1,
                  rides_pp=ride_pp_list)


def test_rider_add_rides_path_not_exist():
    """ Test either if an error is raised when the path is not existing at
    load time. """
    rider = Rider(cyclist_weight=60., max_duration_profile=1)

    assert_raises(ValueError, rider.add_rides_from_path, 'random')


def test_rider_add_rides_path():
    """ Test to check the routine which load file from path. """
    rider = Rider(cyclist_weight=60., max_duration_profile=1)

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_fit = os.path.join(currdir, 'data', 'fit_files')

    rider.add_rides_from_path(path_fit)

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-11-11-39-38.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-07-14-26-22.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-07-26-18-50-56.fit')]

    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=1)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Initialize the rider
    rider2 = Rider(cyclist_weight=60., max_duration_profile=1,
                   rides_pp=ride_pp_list)

    # Check the consistency between the two riders
    for r1, r2 in zip(rider.rides_pp_, rider2.rides_pp_):
        assert_array_equal(r1.data_, r2.data_)


def test_rider_add_rides_path_overwrite():
    """ Test to check the routine which load file from path with overwrite. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-11-11-39-38.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-07-26-18-50-56.fit')]

    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=1)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Initialize the rider
    rider = Rider(cyclist_weight=60., max_duration_profile=1,
                   rides_pp=ride_pp_list)

    # Check the number of ride in the list
    assert_equal(len(rider.rides_pp_), 2)

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_fit = os.path.join(currdir, 'data', 'fit_files')

    rider.add_rides_from_path(path_fit, overwrite=True)
    assert_equal(len(rider.rides_pp_), 3)


def test_rider_add_ride_fit():
    """ Test that the routine to add a ride is working. """
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-11-11-39-38.fit')
    rider.add_ride_from_fit(filename)

        # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-11-11-39-38.fit')]
    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=1)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Initialize the rider
    rider2 = Rider(cyclist_weight=60., max_duration_profile=1,
                   rides_pp=ride_pp_list)

    # Check the consistency
    assert_array_equal(rider.rides_pp_[0].data_, rider2.rides_pp_[0].data_)


def test_rider_delete_ride_no_date():
    """ Test either if an error is raised if the argument passed is not
    a date. """
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-11-11-39-38.fit')
    rider.add_ride_from_fit(filename)
    assert_raises(ValueError, rider.delete_ride, '2014, 05, 11')


def test_rider_delete_ride():
    """ Test that the routine to delete a ride. """
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-11-11-39-38.fit')
    rider.add_ride_from_fit(filename)
    rider.delete_ride(date(2014, 5, 11))

    # Check that there is nothing inside the list
    assert_equal(len(rider.rides_pp_), 0)


def test_rider_delete_ride_warning_nothing():
    """ Test eihter a warning is raised when nothing was deleted. """
    rider = Rider(cyclist_weight=60., max_duration_profile=1)
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-11-11-39-38.fit')
    rider.add_ride_from_fit(filename)
    rider.delete_ride(date(2014, 5, 11))
    assert_warns(UserWarning, rider.delete_ride, date(2014, 5, 11))

    # Check that there is nothing inside the list
    assert_equal(len(rider.rides_pp_), 0)


def test_compute_record_pp():
    """ Test to compute the record power-profile. """
    rider = Rider(cyclist_weight=60., max_duration_profile=1)

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_fit = os.path.join(currdir, 'data', 'fit_files')

    rider.add_rides_from_path(path_fit)
    rider.compute_record_pp()
    data = np.load(os.path.join(currdir, 'data', 'record_pp.npy'))
    # Check that the record is the same that the one previously saved
    assert_array_equal(rider.record_pp_.data_, data)


def test_compute_record_pp_date():
    """ Test to compute the record power-profile with some specific date. """
    rider = Rider(cyclist_weight=60., max_duration_profile=1)

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    path_fit = os.path.join(currdir, 'data', 'fit_files')

    rider.add_rides_from_path(path_fit)
    rider.compute_record_pp(date_start_finish=(date(2014, 5, 7),
                                               date(2014, 5, 11)))
    data = np.load(os.path.join(currdir, 'data', 'record_pp_date.npy'))
    # Check that the record is the same that the one previously saved
    assert_array_equal(rider.record_pp_.data_, data)
