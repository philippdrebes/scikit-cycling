"""
Methods to handle input/output files.
"""

import numpy as np
from os.path import isfile
import warnings

from fitparse import FitFile


def load_power_from_fit(filename):
    """ Method to open the power data from FIT file into a numpy array.

    Parameters
    ----------
    filename: str,
        Path to the FIT file.
    """

    # Check that the filename has the good extension
    if filename.endswith('.fit') is not True:
        raise ValueError('The file does not have the right extension.'
                         ' Expected *.fit.')

    # Check if the file exists
    if isfile(filename) is not True:
        raise ValueError('The file does not exist. Please check the path.')

    # Create an object to open the activity
    activity = FitFile(filename)
    activity.parse()

    # Get only the power records
    records = list(activity.get_messages(name='record'))

    # Append the different values inside a list which will be later
    # converted to numpy array
    power_rec = np.zeros((len(records), ))
    # Go through each record
    # In order to send multiple warnings
    warnings.simplefilter('always', UserWarning)
    warn_sample = 0
    for idx_rec, rec in enumerate(records):
        # Extract only the value regarding the power
        p = rec.get_value('power')
        if p is not None:
            power_rec[idx_rec] = float(p)
        else:
            # We put the value to 0 since that it will not influence
            # the computation of the RPP
            power_rec[idx_rec] = 0.
            # We keep track of the number of inconsitent data
            warn_sample += 1

    # Through a warning if there is no power data found
    if len(records) == warn_sample:
        warnings.warn('This file does not contain any power data.'
                      'Be aware.')

    return power_rec
