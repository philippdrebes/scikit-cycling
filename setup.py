#! /usr/bin/env python
"""A set of python modules for cyclist using powermeters."""

import os
import codecs

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('skcycling', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'scikit-cycling'
DESCRIPTION = 'A set of python modules for cyclist using powermeters.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'G. Lemaitre'
MAINTAINER_EMAIL = 'g.lemaitre58@gmail.com'
URL = 'https://github.com/glemaitre/scikit-cycling'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/glemaitre/scikit-cycling'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'six', 'joblib', 'fitparse']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6']
PACKAGE_DATA = {'skcycling': ['datasets/data/*.fit',
                              'datasets/corrupted_data/*.fit']}
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib',
    ]
}


setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      package_data=PACKAGE_DATA,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
