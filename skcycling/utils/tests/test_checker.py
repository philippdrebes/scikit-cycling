""" Testing the checker methods """

import os
import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from skcycling.utils import check_X
from skcycling.utils import check_float
from skcycling.utils import check_filename_fit


def test_check_x_not_vector():
    """ Test if an error is risen if X is not a vector. """
    assert_raises(ValueError, check_X, np.ones((10, 2)))


def test_check_x_convert_float():
    """ Test if array X is converted into float if the input is not. """
    x_out = check_X(np.random.randint(0, high=100, size=(100, )))
    assert_equal(x_out.dtype, np.float64)


def test_check_x_np_float64():
    """ Test everything goes fine with numpy double. """
    x_out = check_X(np.random.random((100, )))

    assert_equal(x_out.dtype, np.float64)


def test_check_float_convertion():
    """ Test if an integer is converted to float """
    assert_equal(type(check_float(1)), float)


def test_check_float_no_conversion():
    """ Test if a float is not converted when a float is given """
    assert_equal(type(check_float(1.)), float)


def test_check_check_filename_fit_wrong_type():
    """ Test either if an error is raised when the wrong type is given. """
    assert_raises(ValueError, check_filename_fit, 1)


def test_check_filename_fit_not_fit():
    """ Test either an error is raised when the file is not with npy
    extension. """
    assert_raises(ValueError, check_filename_fit, 'file.rnd')


def test_check_filename_fit_not_exist():
    """ Test either if an error is raised when the file is not existing. """
    assert_raises(ValueError, check_filename_fit, 'file.fit')


def test_check_filename_fit():
    """ Test the routine to check the filename is fit. """
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', '2013-04-24-22-22-25.fit')
    my_filename = check_filename_fit(filename)

    assert_equal(my_filename, filename)
