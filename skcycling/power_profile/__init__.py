"""
The :mod:`skcycling.power_profile` module include management function for
record power profile.
"""

from .base_power_profile import BasePowerProfile
from .ride_power_profile import RidePowerProfile
from .record_power_profile import RecordPowerProfile

__all__ = ['BasePowerProfile',
           'RidePowerProfile',
           'RecordPowerProfile']
