""" Helper function to check data conformity
"""

import os
import numpy as np

from datetime import date


def check_X(X):
    """ Private function helper to check if X is of proper size

    Parameters
    ----------
    X : array-like, shape = (data, )
    """

    # Check that X is a numpy vector
    if len(X.shape) is not 1:
        raise ValueError('The shape of X is not consistent.'
                         ' It should be a 1D numpy vector.')

    # Check that X is of type float
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    return X


def check_float(X):
    """ Private function helper to check if the value is a float

    Parameters
    ----------
    X :
        Value to check and convert if not float
    """

    # Check that the value is a float
    if type(X) is not float:
        return np.float(X)
    else:
        return X


def check_filename_fit(filename):
    """ Method to check if the filename corresponds to a fit file.

    Parameters
    ----------
    filename : str
        The fit file to check.

    Return
    ------
    filename : str
        The checked filename.
    """

    # Check that filename is of string type
    if isinstance(filename, basestring):
        # Check that this is a fit file
        if filename.endswith('.fit'):
            # Check that the file is existing
            if os.path.isfile(filename):
                return filename
            else:
                raise ValueError('The file does not exist.')
        else:
            raise ValueError('The file is not an fit file.')
    else:
        raise ValueError('The filename needs to be a string.')


def check_filename_pickle_load(filename):
    """ Method to check if the filename corresponds to a pickle file.

    Parameters
    ----------
    filename : str
        The pickle file to check.

    Return
    ------
    filename : str
        The checked filename.
    """

    # Check that filename is of string type
    if isinstance(filename, basestring):
        # Check that this is a fit file
        if filename.endswith('.p'):
            # Check that the file is existing
            if os.path.isfile(filename):
                return filename
            else:
                raise ValueError('The file does not exist.')
        else:
            raise ValueError('The file is not an pickle `.p` file.')
    else:
        raise ValueError('The filename needs to be a string.')


def check_filename_pickle_save(filename):
    """ Function to check the extension of the pickle file.

    Parameter
    ---------
    filename : str
        The filename which needs to be checked.

    Return
    ------
    filename : str
        The filename which has been checked.
    """
    # Check that the filename is a string
    if isinstance(filename, basestring):
        # Check the extension
        if filename.endswith('.p'):
            return filename
        else:
            raise ValueError('The filename should have a `.p` extension.')
    else:
        raise ValueError('The filename needs to be of type string.')


def check_tuple_date(date_tuple):
    """ Function to check if the date tuple is consistent.

    Parameters
    ----------
    date_tuple : tuple of date, shape (start, finish)
        The tuple to check.

    Return
    ------
    date_tuple : tuple of date, shape (start, finish)
        The validated tuple.
    """
    if isinstance(date_tuple, tuple) and len(date_tuple) == 2:
        # Check that the tuple is of write type
        if isinstance(date_tuple[0],
                      date) and isinstance(date_tuple[1],
                                               date):
            # Check that the first date is earlier than the second date
            if date_tuple[0] < date_tuple[1]:
                return date_tuple
            else:
                raise ValueError('The tuple need to be ordered'
                                 ' as (start, finish).')
        else:
            raise ValueError('Use the class `date` inside the tuple.')
    else:
        raise ValueError('The date are ordered a tuple of'
                         ' date (start, finsih).')
