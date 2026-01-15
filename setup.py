"""
Setup script for IABS Data Synchronizer.

For modern installation, see pyproject.toml.
This file is provided for backward compatibility.
"""

from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
)
