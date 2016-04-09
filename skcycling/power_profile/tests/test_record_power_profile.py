"""Test the Record Power Profile class. """

import os
import numpy as np

from datetime import date

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.power_profile import RidePowerProfile
from skcycling.power_profile import RecordPowerProfile
from skcycling.power_profile.record_power_profile import maximal_mean_power

def test_record_pp_fit():
    """ Test the fit routine to compute the record power-profile. """
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

    # Create an object to handle the record power-profile
    record_pp = RecordPowerProfile(max_duration_profile=1)
    # Find the record_rpp by fitting the list of ride power-profile
    record_pp.fit(ride_pp_list)

    data = np.load(os.path.join(currdir, 'data', 'profile_record_pp_1.npy'))
    assert_array_equal(record_pp.data_, data)
    assert_equal(record_pp.data_norm_, None)
    assert_equal(record_pp.max_duration_profile_, 1)
    assert_equal(record_pp.cyclist_weight_, None)
    assert_equal(record_pp.date_profile_, (date(2014, 05, 07),
                                           date(2014, 07, 26)))


def test_record_pp_fit_w_weight():
    """ Test the fit routine to compute the record power-profile
    with weight. """
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
    ride_pp_list = [RidePowerProfile(max_duration_profile=1,
                                     cyclist_weight=60.)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Create an object to handle the record power-profile
    record_pp = RecordPowerProfile(max_duration_profile=1,
                                   cyclist_weight=60.)
    # Find the record_rpp by fitting the list of ride power-profile
    record_pp.fit(ride_pp_list)

    data = np.load(os.path.join(currdir, 'data', 'profile_record_pp_1.npy'))
    assert_array_equal(record_pp.data_, data)
    assert_array_equal(record_pp.data_norm_, data / 60.)
    assert_equal(record_pp.max_duration_profile_, 1)
    assert_equal(record_pp.cyclist_weight_, 60.)
    assert_equal(record_pp.date_profile_, (date(2014, 05, 07),
                                           date(2014, 07, 26)))


def test_record_pp_fit_date():
    """ Test the fit routine to compute the record power-profile
    with weight and different time range. """
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
    ride_pp_list = [RidePowerProfile(max_duration_profile=1,
                                     cyclist_weight=60.)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Create an object to handle the record power-profile
    record_pp = RecordPowerProfile(max_duration_profile=1,
                                   cyclist_weight=60.)
    # Find the record_rpp by fitting the list of ride power-profile
    record_pp.fit(ride_pp_list, date_profile=(date(2014, 05, 07),
                                              date(2014, 05, 11)))

    data = np.load(os.path.join(currdir, 'data', 'profile_record_pp_2.npy'))
    assert_array_equal(record_pp.data_, data)
    assert_array_equal(record_pp.data_norm_, data / 60.)
    assert_equal(record_pp.max_duration_profile_, 1)
    assert_equal(record_pp.cyclist_weight_, 60.)
    assert_equal(record_pp.date_profile_, (date(2014, 05, 07),
                                           date(2014, 05, 11)))


def test_record_pp_fit_wrong_pp_type():
    """ Test either if an error is raised when no a list is passed instead of
    a list of RidePowerProfile. """
    # Create an object to handle the record power-profile
    record_pp = RecordPowerProfile(max_duration_profile=1,
                                   cyclist_weight=60.)
    assert_raises(ValueError, record_pp.fit, 1)


def test_record_pp_fit_wrong_pp_list_type():
    """ Test either if an error is raised when no a list is passed instead of
    a list of RidePowerProfile. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-07-14-26-22.fit')]
    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=1,
                                     cyclist_weight=60.)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Add a dummy element
    ride_pp_list.insert(1, 1)

    # Create an object to handle the record power-profile
    record_pp = RecordPowerProfile(max_duration_profile=1,
                                   cyclist_weight=60.)
    assert_raises(ValueError, record_pp.fit, ride_pp_list)


def test_record_pp_fit_date_wrong_tuple():
    """ Test either if an error is raised when the tuple to represent the date
    is wrong. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-07-14-26-22.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-11-11-39-38.fit')]

    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=1,
                                     cyclist_weight=60.)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Create an object to handle the record power-profile
    record_pp = RecordPowerProfile(max_duration_profile=1,
                                   cyclist_weight=60.)

    # Raise an error when it is not a tuple
    assert_raises(ValueError, record_pp.fit, ride_pp_list, date_profile=[1, 1])
    # Raise an error when the number inside of the tuple is two large
    assert_raises(ValueError, record_pp.fit, ride_pp_list,
                  date_profile=(1, 2, 3))
    # Raise an error when the value inside the tuple are not from the
    # date class
    assert_raises(ValueError, record_pp.fit, ride_pp_list,
                  date_profile=(1, 2))
    # Raise an error when the ordered of the date is not correct
    assert_raises(ValueError, record_pp.fit, ride_pp_list,
                  date_profile=(date(2014, 1, 2), date(2014, 1, 1)))


def test_record_pp_fit_forget_fitting():
    """ Test either if an error is raised when the ride power-profile were
    not fitted. """
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

    # Create an object to handle the record power-profile
    record_pp = RecordPowerProfile(max_duration_profile=1)
    # Find the record_rpp by fitting the list of ride power-profile
    assert_raises(ValueError, record_pp.fit, ride_pp_list)


def test_record_pp_fit_forget_fitting():
    """ Test either if an error is raised when we did not give any duration. """
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

    # Create an object to handle the record power-profile
    record_pp = RecordPowerProfile(max_duration_profile=None)
    # Find the record_rpp by fitting the list of ride power-profile
    assert_raises(ValueError, record_pp.fit, ride_pp_list)


def test_record_pp_fit_different_max_duration():
    """ Test either if an error is raised when the max duration is different
    from on fit to another one. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-07-14-26-22.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-11-11-39-38.fit')]

    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=i+1)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    # Create an object to handle the record power-profile
    record_pp = RecordPowerProfile(max_duration_profile=1)
    # Find the record_rpp by fitting the list of ride power-profile
    assert_raises(ValueError, record_pp.fit, ride_pp_list)


def test_mmp_wrong_type():
    """ Test either if an error is raised when the type passed to the max mean
    power function is wrong. """
    assert_raises(ValueError, maximal_mean_power, 1)

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    # Create a list of file that we will read
    filename_list = [os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-07-14-26-22.fit'),
                     os.path.join(currdir, 'data',
                                  'fit_files', '2014-05-11-11-39-38.fit')]

    # Create a list of ride power-profile
    ride_pp_list = [RidePowerProfile(max_duration_profile=i+1)
                    for i in range(len(filename_list))]

    # Fit each file of the list
    for ride, filename in zip(ride_pp_list, filename_list):
        ride.fit(filename)

    ride_pp_list.append(1)

    assert_raises(ValueError, maximal_mean_power, ride_pp_list)
