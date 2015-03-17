Coding style
============

* We follow `PEP 8 <http://legacy.python.org/dev/peps/pep-0008/>`_.

* Before you check in code, please run ``pyflakes`` and ``pep8``
  (or ``flake8``) on your source files, and make sure you don't get any
  errors.

* Please don't use backticks, they're deprecated. To format strings, use
  the ``%`` operator or ``str.format``.

* Run tests using ``nosetests``.

* Please don't use ``assert`` in tests. Use the ``numpy.testing`` and
  ``nose.tools`` assertions.
