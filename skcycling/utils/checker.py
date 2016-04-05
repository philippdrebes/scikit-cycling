""" Helper function to check data conformity
"""

import numpy as np
import os


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
