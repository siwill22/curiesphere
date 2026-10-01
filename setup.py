#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

install_requires = ['numpy',
                    'scipy',
                    'matplotlib',
                    'xarray',
                    'pandas',
                    'geopandas',
                    'rasterio',
                    'pyshtools>=4.8.0',
                    'pygmt',
                    'pygplates',
                    'astropy_healpix']

setup(name='curiesphere',
      version='2.0.0',
      description='Forward modelling of lithospheric magnetization using vector spherical harmonics',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/siwill22/curiesphere',
      author='David Gubbins, Jiang Yi, Simon Williams',
      license='MIT',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering'
      ],
      keywords=['magnetic', 'vector spherical harmonics', 'geophysics'],
      packages=find_packages(include=['remit', 'remit.*']),
      package_data={'remit.data': ['*.txt', 'continents/*.nc', 'oceans/*.nc', 'shc/*.cof']},
      install_requires=install_requires,
      python_requires='>=3.8')
