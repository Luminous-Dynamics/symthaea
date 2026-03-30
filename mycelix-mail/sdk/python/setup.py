# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Setup script for Mycelix Mail Python SDK
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mycelix-sdk",
    version="1.0.0",
    author="Mycelix Team",
    author_email="sdk@mycelix.mail",
    description="Official Python SDK for Mycelix Mail API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mycelix/mycelix-mail",
    project_urls={
        "Documentation": "https://mycelix.mail/docs/sdk/python",
        "Bug Tracker": "https://github.com/mycelix/mycelix-mail/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Communications :: Email",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "httpx>=0.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
    },
    keywords=[
        "mycelix",
        "email",
        "mail",
        "api",
        "sdk",
        "trust",
        "web-of-trust",
        "decentralized",
    ],
)
