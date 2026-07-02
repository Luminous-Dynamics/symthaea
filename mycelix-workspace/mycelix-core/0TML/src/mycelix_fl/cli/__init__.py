# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL Command Line Interface

Provides CLI tools for running FL experiments, benchmarks, and attack testing.

Usage:
    mycelix-fl run --config experiment.yaml
    mycelix-fl benchmark --nodes 50 --rounds 10
    mycelix-fl attack-test --scenario cartel

Author: Luminous Dynamics
Date: December 31, 2025
"""

from mycelix_fl.cli.main import main, app

__all__ = ["main", "app"]
