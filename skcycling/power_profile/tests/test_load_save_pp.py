""" Test the loading and saving using pickles. """

import os
import shutil
from tempfile import mkdtemp

import pytest

from numpy.testing import assert_allclose

from skcycling.datasets import load_toy
from skcycling.power_profile import RidePowerProfile


def test_save_load_ride_pp():
    filename = load_toy()[0]
    my_ride_rpp = RidePowerProfile(max_duration_profile=1)
    my_ride_rpp.fit(filename)

    tmp_dir = mkdtemp()
    try:
        store_filename = os.path.join(tmp_dir, 'ride_rpp.pkl')
        my_ride_rpp.save_to_pickles(store_filename)
        obj = RidePowerProfile.load_from_pickles(store_filename)

        assert_allclose(my_ride_rpp.data_, obj.data_)
        assert_allclose(my_ride_rpp.data_norm_, obj.data_norm_)
        assert my_ride_rpp.cyclist_weight == pytest.approx(obj.cyclist_weight)
        assert my_ride_rpp.max_duration_profile == obj.max_duration_profile
        assert my_ride_rpp.date_profile_ == obj.date_profile_
        assert my_ride_rpp.filename_ == obj.filename_
    finally:
        shutil.rmtree(tmp_dir)
