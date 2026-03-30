# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix-DeSci Python SDK Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mycelix-py",
    version="0.1.0",
    author="Luminous Dynamics",
    author_email="contact@luminousdynamics.com",
    description="Python SDK for Mycelix-DeSci decentralized science platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Luminous-Dynamics/mycelix-desci",
    project_urls={
        "Bug Tracker": "https://github.com/Luminous-Dynamics/mycelix-desci/issues",
        "Documentation": "https://github.com/Luminous-Dynamics/mycelix-desci/tree/main/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.0.270",
        ],
        "async": ["httpx[http2]>=0.24.0"],
    },
)
