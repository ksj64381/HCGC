"""
setup.py -- Build hcgc_module C++ extension.

Usage:
    python setup.py build_ext --inplace

Requirements:
    pip install pybind11
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "hcgc_module",
        ["hcgc_module.cpp"],
    ),
]

setup(
    name="hcgc",
    version="0.1.0",
    description="Heterogeneous Graph Coarsening via Coalition Games",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
)
