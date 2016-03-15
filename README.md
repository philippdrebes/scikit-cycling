Scikit-cycling
=========

[![Code Health](https://landscape.io/github/glemaitre/scikit-cycling/master/landscape.svg?style=flat)](https://landscape.io/github/glemaitre/scikit-cycling/master) [![Coverage Status](https://coveralls.io/repos/github/glemaitre/scikit-cycling/badge.svg?branch=master)](https://coveralls.io/github/glemaitre/scikit-cycling?branch=master) [![Build Status](https://travis-ci.org/glemaitre/scikit-cycling.svg?branch=master)](https://travis-ci.org/glemaitre/scikit-cycling) [![Code Issues](https://www.quantifiedcode.com/api/v1/project/ba315c8a63634cd6af7851d88db9fbcc/badge.svg)](https://www.quantifiedcode.com/app/project/ba315c8a63634cd6af7851d88db9fbcc)

#### Manifesto

Because *Human* is **perfectible** and **error-prone**, because *Science* should be **open** and **flow** and because *cogito ergo sum*.

Goal
----

This toolbox will aggregate some useful tools to read into power data acquired by cyclists.

File Structure
--------------

```
.
├── build_tools
│   └── travis
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── setup.cfg
├── setup.py
├── skcycling
│   ├── __init__.py
│   ├── metrics
│   ├── power_profile
│   ├── restoration
│   ├── setup.py
│   └── utils
└── third-party
    └── python-fitparse
```

Installation
------------

### Dependencies

This package needs the following dependencies:

* Numpy,
* Scipy,
* Joblib,
* Fitparse.

The package `fitparse` is part of this repository as a submodule. Follow the cloning and install procedure.

### Cloning

You can clone this repository with the usual `git clone --recursive`.

### Installation

First, install the `fitparse` package attached as a submodule.

```
cd third-party/python-fitparse
python setup.py install
```

Then, go back to the root directory and install `skcycling`.

```
cd ../../
python setup.py install
```
