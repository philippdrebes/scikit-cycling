""" Testing the checker methods """
from datetime import date

import pytest

from skcycling.datasets import load_toy
from skcycling.utils import check_tuple_date
from skcycling.utils import check_filename_fit


@pytest.mark.parametrize(
    "filename,msg",
    [(1, 'filename needs to be a string'),
     ('file.rnd', 'The file is not a fit ride.'),
     ('file.fit', 'The file does not exist.')])
def test_check_check_filename_error(filename, msg):
    with pytest.raises(ValueError, message=msg):
        check_filename_fit(filename)


def test_check_filename_fit():
    filename = load_toy()[0]
    my_filename = check_filename_fit(filename)
    assert my_filename == filename


@pytest.mark.parametrize(
    "value",
    [(1),
     ((1, 1, 1)),
     (1, 1),
     (date(2014, 1, 1), date(2013, 1, 1))])
def test_tuple_date_error(value):
    with pytest.raises(ValueError):
        check_tuple_date(value)


def test_tuple_date():
    dt = (date(2014, 1, 1), date(2015, 1, 1))
    dt_out = check_tuple_date(dt)

    assert dt_out == dt
