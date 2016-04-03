"""Test the power profile class."""

import os
import numpy as np

from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal

from skcycling.utils import load_power_from_fit

from skcycling.power_profile import Rpp
from skcycling.power_profile import compute_ride_rpp
from skcycling.power_profile import _rpp_parallel


DECIMAL_PRECISION = 3


def test_rpp_load_wrong_filename():
    """ Test either if an error is raised when the file is not an npy file. """

    # Try a non npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    my_rpp = Rpp()
    assert_raises(ValueError, my_rpp.load_from_npy, filename)


def test_rpp_load_no_file():
    """ Test either if an error is raised when the file does not exist. """

    # Create an unknown file
    filename = 'none.npy'
    my_rpp = Rpp()
    assert_raises(ValueError, my_rpp.load_from_npy, filename)


def test_rpp_load_wrong_weight():
    """ Test either if an error is raised if the type for cyclist weight
    is wrong. """

    # Create the path to the profile file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    assert_raises(ValueError, my_rpp.load_from_npy, filename,
                  cyclist_weight='wrong')


def test_rpp_load_no_weight():
    """ Test the routine to read the Rpp with no weight. """

    # Create the path to the profile file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename)

    # Check if the value are ok
    data = np.load(filename)
    assert_equal(my_rpp.rpp_, data)
    assert_equal(my_rpp.rpp_norm_, None)
    assert_equal(my_rpp.max_duration_rpp_, 300)
    assert_equal(my_rpp.cyclist_weight_, None)


def test_rpp_load_weight():
    """ Test the routine to read the Rpp with given weight. """

    # Create the path to the profile file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename, cyclist_weight=60.)

    # Check if the value are ok
    data = np.load(filename)
    assert_equal(my_rpp.rpp_, data)
    filename = os.path.join(currdir, 'data', 'profile_norm.npy')
    data = np.load(filename)
    assert_equal(my_rpp.rpp_norm_, data)
    assert_equal(my_rpp.max_duration_rpp_, 300)
    assert_equal(my_rpp.cyclist_weight_, 60.)


def test_fit_no_duration():
    """ Test if an error is raised when the no fitting is called and
    no duration is provided. """

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec = load_power_from_fit(filename)

    # Compute the rpp not in parallel
    my_rpp = Rpp()
    assert_raises(ValueError, my_rpp.fit, power_rec)


def test_compute_ride_rpp_not_par():
    """ Test the routine to compute the ride Rpp not in parallel. """

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec = load_power_from_fit(filename)

    # Compute the rpp not in parallel
    rpp_ride = compute_ride_rpp(power_rec, 10, in_parallel=False)

    filename = os.path.join(currdir, 'data', 'profile_ride.npy')
    data = np.load(filename)
    assert_equal(rpp_ride, data)


def test_compute_ride_rpp_par():
    """ Test the routine to compute the ride Rpp in parallel. """

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec = load_power_from_fit(filename)

    # Compute the rpp not in parallel
    rpp_ride = compute_ride_rpp(power_rec, 10, in_parallel=True)

    filename = os.path.join(currdir, 'data', 'profile_ride.npy')
    data = np.load(filename)
    assert_equal(rpp_ride, data)


def test_rpp_parallel():
    """ Test the best mean function. """

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec = load_power_from_fit(filename)

    # Compute the rpp for one specific value
    val = _rpp_parallel(power_rec, 1)
    assert_equal(val, 500.)


def test_rpp_fit():
    """ Test if the rpp is computed properly. """

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec = load_power_from_fit(filename)

    # Create an Rpp object
    my_rpp = Rpp(max_duration_rpp=1)
    my_rpp.fit(power_rec)

    data = np.load(os.path.join(currdir, 'data', 'profile_fit_1.npy'))
    assert_array_equal(my_rpp.rpp_, data)


def test_rpp_partial_fitting():
    """ Test the partial fitting with update of the rpp. """

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec = load_power_from_fit(filename)

    # Create an Rpp object
    my_rpp = Rpp(max_duration_rpp=1)
    my_rpp.fit(power_rec)

    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')
    power_rec = load_power_from_fit(filename)
    my_rpp.partial_fit(power_rec)

    data = np.load(os.path.join(currdir, 'data', 'profile_fit_2.npy'))
    assert_array_equal(my_rpp.rpp_, data)


def test_rpp_partial_fitting_weight():
    """ Test the partial fitting with update of the rpp with weight. """

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec = load_power_from_fit(filename)

    # Create an Rpp object
    my_rpp = Rpp(max_duration_rpp=1, cyclist_weight=60.)
    my_rpp.fit(power_rec)

    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')
    power_rec = load_power_from_fit(filename)
    my_rpp.partial_fit(power_rec)

    data = np.load(os.path.join(currdir, 'data', 'profile_fit_2.npy'))
    assert_array_equal(my_rpp.rpp_, data)


def test_rpp_partial_fitting_refit():
    """ Test the partial fitting with update of the rpp with a refit. """

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec_1 = load_power_from_fit(filename)

    # Create an Rpp object
    my_rpp = Rpp(max_duration_rpp=1)
    my_rpp.fit(power_rec_1)

    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')
    power_rec_2 = load_power_from_fit(filename)
    my_rpp.partial_fit(power_rec_2)
    my_rpp.partial_fit(power_rec_1, refit=True)

    data = np.load(os.path.join(currdir, 'data', 'profile_fit_1.npy'))
    assert_array_equal(my_rpp.rpp_, data)


def test_rpp_fit_weight():
    """ Test if the rpp is computed properly with cyclist weight. """

    # Create the path to read the npy file
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data',
                            'fit_files', '2014-05-07-14-26-22.fit')

    # Load the fit file
    power_rec = load_power_from_fit(filename)

    # Create an Rpp object
    my_rpp = Rpp(max_duration_rpp=1, cyclist_weight=60.)
    my_rpp.fit(power_rec)

    data = np.load(os.path.join(currdir, 'data', 'profile_fit_1.npy'))
    assert_array_equal(my_rpp.rpp_, data)
    data = np.load(os.path.join(currdir, 'data', 'profile_fit_weight_1.npy'))
    assert_array_equal(my_rpp.rpp_norm_, data)


def test_resampling_wrong_weight():
    """ Test either if an error is raised when normalized data are requested
    without providing cyclist weight. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename)
    assert_raises(ValueError, my_rpp.resampling_rpp, 10, normalized=True)


def test_resampling_rpp():
    """ Check that the resampling is working. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename)
    val = my_rpp.resampling_rpp(10)
    assert_almost_equal(val, 401.7436556297533, decimal=DECIMAL_PRECISION)


def test_resampling_rpp_weight():
    """ Check that the resampling is working with normalized data. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename, cyclist_weight=60.)
    val = my_rpp.resampling_rpp(10, normalized=True)
    assert_almost_equal(val, 6.695727593829221, decimal=DECIMAL_PRECISION)


def test_aerobic_meta_model_wrong_method():
    """ Test either if an error is raised when the optimization method
    is unknown. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename)
    assert_raises(my_rpp.aerobic_meta_model, method='None')


def test_aerobic_meta_model_no_normalized():
    """ Test either if an error is raised when normalized data are requisted
    without providing cyclist weight. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename)
    assert_raises(my_rpp.aerobic_meta_model, normalized=True)


def test_aerobic_meta_model():
    """ Test the aerobic metabolism model using default parameters. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename)
    pma, t_pma, aei = my_rpp.aerobic_meta_model()

    # Check the different value
    assert_almost_equal(pma, 453.37229888268155,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(t_pma, 3.,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(aei, -11.487957337374663,
                        decimal=DECIMAL_PRECISION)


def test_aerobic_meta_model_ts():
    """ Test the aerobic metabolism model using default parameters with
    a given ts. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename)
    ts_reg = np.array([3., 4., 5., 6., 7., 10, 20, 30, 45, 60, 120, 180, 240])
    pma, t_pma, aei = my_rpp.aerobic_meta_model(ts=ts_reg)

    # Check the different value
    assert_almost_equal(pma, 453.37229888268155,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(t_pma, 3.,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(aei, -11.497339815062894,
                        decimal=DECIMAL_PRECISION)


def test_aerobic_meta_model_weight():
    """ Test the aerobic metabolism model using default parameters with
    cyclist weight information. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename, cyclist_weight=60.)
    pma, t_pma, aei = my_rpp.aerobic_meta_model(normalized=True)

    # Check the different value
    assert_almost_equal(pma, 7.5562049813780261,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(t_pma, 3.,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(aei, -11.487957337374663,
                        decimal=DECIMAL_PRECISION)


def test_aerobic_meta_model_lm():
    """ Test the aerobic metabolism model using default parameters and
    Levevenberg Marquardt. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename)
    pma, t_pma, aei = my_rpp.aerobic_meta_model(method='lm')

    # Check the different value
    assert_almost_equal(pma, 453.37229888268155,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(t_pma, 3.,
                        decimal=DECIMAL_PRECISION)
    assert_almost_equal(aei, -11.487957337374663,
                        decimal=DECIMAL_PRECISION)


def test_aerobic_meta_model_unknown_method():
    """ Test either if an error if the method used during
    fitting is unknown. """

    # Read the profile data
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', 'profile.npy')

    # Read the profile
    my_rpp = Rpp()
    my_rpp.load_from_npy(filename)
    assert_raises(NotImplementedError, my_rpp.aerobic_meta_model,
                  method='None')
