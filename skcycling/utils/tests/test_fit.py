""" Test the functions for mathematical fitting. """

import pytest

import numpy as np
from numpy.testing import assert_allclose

from skcycling.utils import res_std_dev
from skcycling.utils import r_squared
from skcycling.utils import log_linear_fitting
from skcycling.utils import linear_model
from skcycling.utils import log_linear_model


def test_res_std_dec_wrong_sz():
    """ Test either if an error is raised if the estimate and model sizes
    are not consistent. """
    model = np.arange(10)
    estimate = np.ones((5, )) * 5.

    with pytest.raises(ValueError):
        res_std_dev(model, estimate)


def test_res_std_dec():
    """ Test the function which compute the residual standard deviation. """
    model = np.arange(10)
    estimate = np.ones((10, )) * 5.

    assert res_std_dev(model, estimate) == pytest.approx(3.2596012026)


def test_r_squared_wrong_sz():
    """ Test either if an error is raised if the estimate and model sizes
    are not consistent. """
    model = np.arange(10)
    estimate = np.ones((5, )) * 5.

    with pytest.raises(ValueError):
        r_squared(model, estimate)


def test_r_squared():
    """ Test the function which compute the coefficient of determination. """
    model = np.arange(10)
    estimate = np.ones((10, )) * 5.
    assert r_squared(model, estimate) == pytest.approx(-0.030303030303030276)


def test_linear_model():
    """ Test the linear model routine. """
    assert linear_model(3., 4., 2.) == pytest.approx(14.)


def test_log_linear_model():
    """ Test the linear model routine. """
    assert log_linear_model(3., 4., 2.) == pytest.approx(6.3944491546724391)


def test_log_linear_fitting_wrong_size():
    """ Test either if an error is raised when the size of x and y are
    not the same. """
    x = np.arange(2., 10.)
    y = np.arange(2., 9.)

    with pytest.raises(ValueError):
        log_linear_fitting(x, y)


def test_log_linear_fitting_lsq():
    """ Test either if an error is raised when the size of x and y are
    not the same. """
    x = np.arange(3, 10, dtype=float)
    y = linear_model(x, 2., 3.)

    method = 'lsq'
    slope, intercept, std_err, coeff_det = log_linear_fitting(x, y, method)

    assert slope == pytest.approx(10.884469280692462)
    assert intercept == pytest.approx(-3.8280798214097231)
    assert std_err == pytest.approx(0.74166428170326482)
    assert coeff_det == pytest.approx(0.97544348630560629)


def test_log_linear_fitting_lm():
    """ Test either if an error is raised when the size of x and y are
    not the same. """
    x = np.arange(3, 10, dtype=float)
    y = linear_model(x, 2., 3.)

    method = 'lm'
    slope, intercept, std_err, coeff_det = log_linear_fitting(x, y, method)

    assert slope == pytest.approx(10.884469280692462)
    assert intercept == pytest.approx(-3.8280798214097231)
    assert std_err == pytest.approx(0.74166428170326482)
    assert coeff_det == pytest.approx(0.97544348630560629)
