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
This makes uses of the python bindings of the Point Cloud Library (PCL).

Copyright 2014-2015 Netherlands eScience Center. Covered by the Apache 2.0
License, see the file ``LICENSE.txt``.

Installing
----------

First install the required dependencies:

* Python 2.7
* NumPy
* SciPy 
* virtualenv
* LibLAS
* PCL 1.7

Now set up an environment::

    $ virtualenv /some/where/env --system-site-packages
    $ . /some/where/env/activate


Install the python packages listed in ``requirements.txt`` using pip::

    $ pip install -r requirements.txt
    $ pip install -U nose  # make sure nose runs in the virtualenv
    $ python setup.py install

To exit the python virtualenv run::

    $ deactivate

Testing
-------

To run unit tests, issue::

    $ nosetests

Documentation
-------------

Documentation can be found here_

.. _here: http://nlesc.github.io/PattyAnalytics/
