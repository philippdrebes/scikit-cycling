""" Testing the input/output methods for FIT files """

import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from numpy.testing import assert_warns

import warnings
import os

from skcycling.utils import load_power_from_fit


def test_load_power_not_fit():
    """ Test if an error is risen in case that the file is not a FIT file. """

    filename = 'example.txt'

    assert_raises(ValueError, load_power_from_fit, filename)


def test_load_power_check_file_exist():
    """ Test if an error is risen if the FIT file does not exist. """

    filename = 'example.fit'

    assert_raises(ValueError, load_power_from_fit, filename)


def test_load_power_if_no_power():
    """ Test if a warning if raise if there is no power data. """

    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(currdir, 'data', '2014-05-17-10-44-53.fit')

    assert_warns(UserWarning, load_power_from_fit, filename)
