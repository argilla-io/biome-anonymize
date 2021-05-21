#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob

from setuptools import setup

try:
    from setuptools import find_namespace_packages
except ImportError as error:
    raise ImportError("Make sure you have setuptools >= 40.1.0 installed!") from error


if __name__ == "__main__":
    setup(
        name="anonymizer_recognai",
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
        install_requires=[
            "presidio-analyzer==2.0.1",
            "presidio-anonymizer==2.0.1",
            "pandas~=1.1.5"
        ],
    )
