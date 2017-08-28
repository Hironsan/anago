#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import sys

from setuptools import find_packages, setup

from pypandoc import convert

# Package meta-data.
NAME = 'anago'
DESCRIPTION = 'Sequence labeling library using Keras.'
URL = 'https://github.com/Hironsan/anago'
EMAIL = 'hiroki.nakayama.py@gmail.com'
AUTHOR = 'Hironsan'
LICENSE = 'MIT'

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + convert(f, 'rst')

about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist bdist_wheel upload')
    sys.exit()

required = [
    'keras', 'h5py', 'scikit-learn', 'numpy'
]

setup(
    name=NAME,
    version=about['__version__'],
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