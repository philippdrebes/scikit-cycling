""" Test the loading and saving using pickles. """

import os
import shutil

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from skcycling.data_management import Rider
from skcycling.power_profile import RidePowerProfile
from skcycling.power_profile import RecordPowerProfile


def test_save_load_rider():
    """ Test the routine to save the object RidePowerProfile. """
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
    rider.compute_record_pp()

    # Save the object
    store_filename = os.path.join(currdir, 'data', 'subdir', 'ride_rpp.p')
    rider.save_to_pickles(store_filename)

    # Save the object
    store_filename = os.path.join(currdir, 'data', 'rider.p')
    rider.save_to_pickles(store_filename)

    # Try to load the file
    obj = Rider.load_from_pickles(store_filename)

    # Check that the object have the same values
    assert_array_equal(rider.rides_pp_[0].data_, obj.rides_pp_[0].data_)
    assert_array_equal(rider.record_pp_.data_, obj.record_pp_.data_)
    assert_equal(rider.max_duration_profile_, obj.max_duration_profile_)
    assert_equal(rider.cyclist_weight_, obj.cyclist_weight_)

    # Remove the created directory
    shutil.rmtree(os.path.join(currdir, 'data', 'subdir'))
