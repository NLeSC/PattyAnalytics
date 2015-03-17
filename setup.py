from distutils.core import setup

setup(name='patty',
      description='pointcloud registration, segmentation, and LAS loading',
      url='http://github.com/NLeSC/PattyAnalytics',
      version='0.1',
      author='NLeSC analytics',
      author_email='j.borgorff@esciencecenter.nl',
      license='Apache 2.0',
      packages=["patty", "patty.registration", "patty.segmentation"],
      )
