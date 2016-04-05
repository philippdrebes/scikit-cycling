import os

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_raises

from skcycling.power_profile import RidePowerProfile
from skcycling.power_profile import RecordPowerProfile


PRECISION_TEST = 3


def test_save_load_ride_pp():
    """ Test the routine to save the object RidePowerProfile. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-07-14-26-22.fit')

    my_ride_rpp = RidePowerProfile(max_duration_profile=1)
    my_ride_rpp.fit(filename)

    # Now that the processing is done let's make a resampling
    val = my_ride_rpp.resampling_rpp(.5)
    assert_almost_equal(val, 376.9, decimal=PRECISION_TEST)


def test_save_load_ride_pp_weight():
    """ Test the routine to save the object RidePowerProfile with a
    given weight. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-07-14-26-22.fit')

    my_ride_rpp = RidePowerProfile(max_duration_profile=1,
                                   cyclist_weight=60.)
    my_ride_rpp.fit(filename)

    # Now that the processing is done let's make a resampling
    val = my_ride_rpp.resampling_rpp(.5, normalized=True)
    assert_almost_equal(val, 6.281666666666666, decimal=PRECISION_TEST)


def test_save_load_ride_pp_wrong_norm():
    """ Test either if an error is raised when the weight was not specified
    normalized data are required. """
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'fit_files',
                            '2014-05-07-14-26-22.fit')

    my_ride_rpp = RidePowerProfile(max_duration_profile=1)
    my_ride_rpp.fit(filename)

    # Now that the processing is done let's make a resampling
    assert_raises(ValueError, my_ride_rpp.resampling_rpp, .5, normalized=True)
