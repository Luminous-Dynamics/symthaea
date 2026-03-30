# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Holochain - Decentralized Federated Learning with Economic Incentives

A fully decentralized P2P federated learning system using Holochain DHT for coordination
and ZeroTrustML Credits for economic incentives and Byzantine resistance.
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    version_file = os.path.join("src", "zerotrustml", "__init__.py")
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Read long description from README
def get_long_description():
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return __doc__

setup(
    name="zerotrustml-holochain",
    version=get_version(),
    author="Luminous Dynamics",
    author_email="dev@luminousdynamics.org",
    description="Decentralized Federated Learning with Economic Incentives",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/luminous-dynamics/0TML",
    project_urls={
        "Bug Tracker": "https://github.com/luminous-dynamics/0TML/issues",
        "Documentation": "https://zerotrustml.readthedocs.io",
        "Source Code": "https://github.com/luminous-dynamics/0TML",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "asyncio>=3.4.3",
        "websockets>=11.0",
        "aiohttp>=3.8.0",
        "cryptography>=41.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "ruff>=0.0.290",
        ],
        "holochain": [
            # Optional: Only needed if running real Holochain conductor
            # Otherwise uses mock mode
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "zerotrustml=zerotrustml.cli:main",
            "zerotrustml-node=zerotrustml.node_cli:main",
            "zerotrustml-monitor=zerotrustml.monitoring.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "zerotrustml": ["py.typed"],
    },
    zip_safe=False,
    keywords=[
        "federated-learning",
        "machine-learning",
        "distributed-systems",
        "holochain",
        "blockchain",
        "privacy",
        "byzantine-resistance",
        "decentralized",
        "p2p",
    ],
)
