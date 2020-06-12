#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, Extension, find_packages
except ImportError:
    from distutils.core import setup, Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize
import numpy


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()

test_requirements = [
    # TODO: put package test requirements here
]

ext_modules = [
    Extension("src.algorithms.common.cython.cyt_dataset",
              ["src/algorithms/common/cython/cyt_dataset.pyx"]),
              #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
    Extension("src.algorithms.common.cython.cyt_gp",
              ["src/algorithms/common/cython/cyt_gp.pyx"]),
              #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
    Extension("src.algorithms.common.cython.cyt_fuzzy_mf_v2",
              ["src/algorithms/common/cython/cyt_fuzzy_mf_v2.pyx"]),
    Extension("src.algorithms.ant_colony.cython.cyt_aco_grad",
              ["src/algorithms/ant_colony/cython/cyt_aco_grad.pyx"]),
              #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
]
# ext_modules = cythonize("src/algorithms/common/cyt_dataset.pyx", annotate=True)

setup(
    name='aco-grad',
    version='2.2',
    description="A Python implementation of ant colony optimization of "
                "the GRAANK algorithm.",
    long_description=readme + '\n\n' + history,
    author="Dickson Owuor",
    author_email='owuordickson@ieee.org',
    url='https://github.com/owuordickson/ant-colony-gp',
    packages=find_packages(),
    # package_dir={'src': 'src'},
    # packages="src",
    include_package_data=True,
    # package_data={'src/algorithms/common/cython': ['*.pxd']},
    install_requires=requirements,
    # cmdclass={'build_ext': build_ext},
    # ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    license="MIT",
    zip_safe=False,
    keywords='aco, graank, gradual patterns',
    classifiers=[
        'Development Status :: 1 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Massachusetts Institute of Technology (MIT) '
        'License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
