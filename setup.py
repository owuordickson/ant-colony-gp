#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
from Cython.Build import cythonize

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()

test_requirements = [
    # TODO: put package test requirements here
]

modules = cythonize("src/algorithms/handle_data/cyt_handle_data.pyx", annotate=True)


setup(
    name='aco-grad',
    version='2.0',
    description="A Python implementation of ant colony optimization of the GRAANK algorithm.",
    long_description=readme + '\n\n' + history,
    author="Dickson Owuor",
    author_email='owuordickson@ieee.org',
    url='https://github.com/owuordickson/ant-colony-gp',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    ext_modules=modules,
    license="MIT",
    zip_safe=False,
    keywords='aco, graank, gradual patterns',
    classifiers=[
        'Development Status :: 1 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Massachusetts Institute of Technology (MIT) License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
