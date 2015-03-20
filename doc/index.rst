.. Patty Analytics documentation master file, created by
   sphinx-quickstart on Wed Mar 18 10:04:17 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Patty Analytics
===============

Contents:

.. toctree::
   :maxdepth: 2

   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Running scripts
===============

** Registration script **

The core functionality of the PattyAnalytics project is the registration of dense point clouds on a reference 'drive map'. This is done through the registration script: scripts/registration.py

This script can be run as follows:

```
Usage:
  registration.py [-h] [-d <sample>] [-u <upfile>] <source> <drivemap> <footprint> <output>

Positional arguments:
  source       Source LAS file
  drivemap     Target LAS file to map source to
  footprint    Footprint for the source LAS file
  output       file to write output LAS to
```

Example run:
```
python scripts/registration.py data/SOURCE/SITE_X.las data/DRIVEMAP/X.las data/FOOTPRINT/X.footprint.csv data/OUTPUT/SITE_X.las -d 100000
```
