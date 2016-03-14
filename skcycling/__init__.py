"""Cycling Processing Toolbox (Toolbox for SciPy)

``scikit-cycling`` (a.k.a ``skcycling``) is a set of python methods to analyse file
extracted from powermeters.

Subpackages
-----------
metrics
    Metrics to quantify cyclist ride.
power_profile
    Record power-profile of cyclist.
restoration
    Utility for denoising cyclist ride.
utils
    Utility to read and save cycling ride.
"""

import os.path as osp
import imp
import functools
import warnings
import sys

__version__ = '0.1dev'

pkg_dir = osp.abspath(osp.dirname(__file__))

try:
    imp.find_module('nose')
except ImportError:
    def _test(doctest=False, verbose=False):
        """This would run all unit tests, but nose couldn't be
        imported so the test suite can not run.
        """
        raise ImportError("Could not load nose. Unit tests not available.")

else:
    def _test(doctest=False, verbose=False):
        """Run all unit tests."""
        import nose
        args = ['', pkg_dir, '--exe', '--ignore-files=^_test']
        if verbose:
            args.extend(['-v', '-s'])
        if doctest:
            args.extend(['--with-doctest', '--ignore-files=^\.',
                         '--ignore-files=^setup\.py$$', '--ignore-files=test'])
            # Make sure warnings do not break the doc tests
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                success = nose.run('skcycling', argv=args)
        else:
            success = nose.run('skcycling', argv=args)
        # Return sys.exit code
        if success:
            return 0
        else:
            return 1

# do not use `test` as function name as this leads to a recursion problem with
# the nose test suite
test = _test
test_verbose = functools.partial(test, verbose=True)
test_verbose.__doc__ = test.__doc__
doctest = functools.partial(test, doctest=True)
doctest.__doc__ = doctest.__doc__
doctest_verbose = functools.partial(test, doctest=True, verbose=True)
doctest_verbose.__doc__ = doctest.__doc__


# Logic for checking for improper install and importing while in the source
# tree when package has not been installed inplace.
# Code adapted from scikit-learn's __check_build module.
_INPLACE_MSG = """
It appears that you are importing a local scikit-cycling source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""

_STANDARD_MSG = """
Your install of scikit-image appears to be broken. """


def _raise_build_error(e):
    # Raise a comprehensible error
    local_dir = osp.split(__file__)[0]
    msg = _STANDARD_MSG
    if local_dir == "skcycling":
        # Picking up the local install: this will work only if the
        # install is an 'inplace build'
        msg = _INPLACE_MSG
    raise ImportError("""%s
It seems that scikit-image has not been built correctly.
%s""" % (e, msg))

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __SKCYCLING_SETUP__
except NameError:
    __SKCYCLING_SETUP__ = False

if __SKCYCLING_SETUP__:
    sys.stderr.write('Partial import of skimage during the build process.\n')
    # We are not importing the rest of the scikit during the build
    # process, as it may not be compiled yet
else:
    try:
        from power_profile import Rpp
        del Rpp
    except ImportError as e:
        _raise_build_error(e)

if sys.version.startswith('2.6'):
    msg = ("Python 2.6 is deprecated and will not be supported in "
           "scikit-cycling 0.1+")
    warnings.warn(msg, stacklevel=2)

del warnings, functools, osp, imp, sys
