|Travis|_ |Quality-score|_ |Coverage|_

.. |Travis| image:: https://api.travis-ci.org/NLeSC/PattyAnalytics.png?branch=master
.. _Travis: https://travis-ci.org/NLeSC/PattyAnalytics

.. |Quality-score| image:: https://scrutinizer-ci.com/g/NLeSC/PattyAnalytics/badges/quality-score.png?b=master
.. _Quality-score: https://scrutinizer-ci.com/g/NLeSC/PattyAnalytics/

.. |Coverage| image:: https://scrutinizer-ci.com/g/NLeSC/PattyAnalytics/badges/coverage.png?b=master
.. _Coverage: https://scrutinizer-ci.com/g/NLeSC/PattyAnalytics/

Patty Analytics
===============

Reusable point cloud analytics software. Includes segmentation, registration,
file format conversion.

Copyright 2014-2015 Netherlands eScience Center. Covered by the Apache 2.0
License, see the file ``LICENSE.txt``.

Installing
----------

First install the required dependencies:

* Python 2.7
* NumPy
* SciPy
* LibLAS
* PCL 1.7
* virtualenv

Now set up an environment::

    virtualenv /some/where/env
    pip install -r requirements.txt
    python setup.py install

Testing
-------

To run unit tests, issue::

    $ nosetests
