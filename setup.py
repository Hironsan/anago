#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

from setuptools import find_packages, setup
# from m2r import parse_from_file

# Package meta-data.
NAME = 'anago'
DESCRIPTION = 'Sequence labeling library using Keras.'
URL = 'https://github.com/Hironsan/anago'
EMAIL = 'hiroki.nakayama.py@gmail.com'
AUTHOR = 'Hironsan'
LICENSE = 'MIT'

here = os.path.abspath(os.path.dirname(__file__))

# long_description = parse_from_file(os.path.join(here, 'README.md'))
long_description = open(os.path.join(here, 'README.md')).read()

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist bdist_wheel upload')
    sys.exit()

required = [
    'Keras>=2.1.6', 'h5py>=2.7.1', 'scikit-learn>=0.19.1',
    'numpy>=1.14.3', 'tensorflow>=1.8.0', 'requests>=2.18.4',
    'seqeval>=0.0.3'
]

setup(
    name=NAME,
    version='1.0.6',
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=required,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
