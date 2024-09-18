#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import versioneer

versioneer.versionfile_source = 'vh0/_version.py'
versioneer.versionfile_build = 'vh0/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'vh0-'

# Convert markdown README.md to restructured text (.rst) for PyPi
'''
try:
    import pypandoc
    rst = pypandoc.convert_file('README.md', 'rst')
    long_description = rst.split('\n', 5)[5]
except(IOError, ImportError):
    print('*** pypandoc is not installed. PYPI description will not be '
          'formatted correctly. ***')
    long_description = open('README.md').read()
'''

install_requires = ['pyshtools>=4.8.0']

setup(name='vh0',
      #version=versioneer.get_version(),
      #cmdclass=versioneer.get_cmdclass(),
      description='vector spherical harmonics analysis of planetary lithospheric magnetic fields',
      #long_description=long_description,
      url='https://github.com/siwill22/vh0',
      authors='David Gubbins, Jiang Yi, Simon Williams',
      #author_email='mark.a.wieczorek@gmail.com',
      #license='BSD',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          #'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering'
      ],
      keywords=['magnetic', 'vector spherical harmonics', 'geophysics'],
      packages=find_packages(),
      include_package_data=True,
      install_requires=install_requires,
      python_requires='>=3.5')