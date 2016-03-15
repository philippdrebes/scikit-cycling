""" Testing the checker methods """

import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.utils import _check_X
from skcycling.utils import _check_float


def test_check_X_not_vector():
    """ Test if an error is risen if X is not a vector. """

    assert_raises(ValueError, _check_X, np.ones((10, 2)))


def test_check_X_convert_float():
    """ Test if array X is converted into float if the input is not. """

    X_out = _check_X(np.random.randint(0, high=100, size=(100, )))
    assert_equal(X_out.dtype, np.float64)


def test_check_X_np_float64():
    """ Test everything goes fine with numpy double. """

    X_out = _check_X(np.random.random((100, )))

    assert_equal(X_out.dtype, np.float64)


def test_check_float_convertion():
    """ Test if an integer is converted to float """

    assert_equal(type(_check_float(1)), float)


def test_check_float_no_conversion():
    """ Test if a float is not converted when a float is given """

    assert_equal(type(_check_float(1.)), float)
