|Travis|_ |Quality-score|_ |Coverage|_

.. |Travis| image:: https://api.travis-ci.org/NLeSC/PattyAnalytics.png?branch=regpipe
.. _Travis: https://travis-ci.org/NLeSC/PattyAnalytics

.. |Quality-score| image:: https://scrutinizer-ci.com/g/NLeSC/PattyAnalytics/badges/quality-score.png?b=regpipe
.. _Quality-score: https://scrutinizer-ci.com/g/NLeSC/PattyAnalytics/

.. |Coverage| image:: https://scrutinizer-ci.com/g/NLeSC/PattyAnalytics/badges/coverage.png?b=regpipe
.. _Coverage: https://scrutinizer-ci.com/g/NLeSC/PattyAnalytics/

Patty Analytics
===============

Reusable point cloud analytics software. Includes segmentation, registration,
file format conversion. This makes uses of the
(python bindings)[https://github.com/NLeSC/python-pcl]  of the
Point Cloud Library (PCL).

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

Running
-------
The main functionality of PatTy Analytics is contained in the **registration**
script. This script takes an unaligned dense point cloud and attempts to
align in on to the existing drivemap. The script also attempts to find the
optimal scaling, rotation and orientation of the dense point cloud, as part of
the alignment process. This script can be run as follows::

    $ python scripts/registration.py SOURCE.las DRIVEMAP.las FOOTPRINT.csv OUTPUT.las

where:

  - *SOURCE.las* -- is the dense point cloud to be registered
  - *DRIVEMAP.las* -- is the drive map where the point cloud is registered to.
  - *FOOTPRINT.csv* -- is the footprint of the point cloud on the drive map.
  - *OUTPUT.las* -- is the resulting registered point cloud.

additionally, an *upfile.json* containing the up vector (estimated from the
camera position) can be provided.

    $ python scripts/registration.py SOURCE.las DRIVEMAP.las FOOTPRINT.csv OUTPUT.las -u upfile.json

## Examples

The following image is a screenshot of a dense point cloud to be registered
on the drive map -- this would correspond to *SOURCE.las*.

![Site 558 dense point cloud](/img/site558_dense.png?raw=true "Dense point cloud")

This screenshot shows the drive map where we want to register to -- this corresponds
to *DRIVEMAP.las*

![Site 558 drive map point cloud](/img/site558_drivemap.png?raw=true "Drive map")

Finally, this screenshot shows the dense point cloud registered on the drive map.
The dense point cloud has been rotated, scaled and translated to find it's best
fit on the drive map -- this corresponds to *OUTPUT.las*.


![Site 558 registered point cloud](/img/site558_registered.png?raw=true "Registered point cloud")

Testing
-------

To run unit tests, issue::

    $ nosetests

Documentation
-------------

API documentation can be found here_

.. _here: http://nlesc.github.io/PattyAnalytics/
