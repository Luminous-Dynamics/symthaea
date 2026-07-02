#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Setup script for Mycelix-DeSci ML components"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mycelix-desci-ml",
    version="0.1.0",
    author="Mycelix Contributors",
    description="Machine Learning and Federated Learning for Mycelix-DeSci",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luminousdynamics/mycelix-desci",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "flower>=1.6.0",
        "biopython>=1.81",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
)
