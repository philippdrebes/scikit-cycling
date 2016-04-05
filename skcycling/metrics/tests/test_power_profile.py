"""Test the metrics linked to the power profile."""

import os

from numpy.testing import assert_raises


from skcycling.power_profile import RidePowerProfile
from skcycling.metrics import aerobic_meta_model


def test_amm_wrong_profile():
    """ Test either if an error is raised when the class of the profile is
    incorrect. """
    assert_raises(ValueError, aerobic_meta_model, 1)


def test_amm():
    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            '2015-02-26-09-08-31.fit')

    my_ride_rpp = RidePowerProfile(max_duration_profile=240, cyclist_weight=60.)
    my_ride_rpp.fit(filename)
    my_ride_rpp.save_to_pickles(os.path.join(currdir, 'data', 'ride_profile.p'))
