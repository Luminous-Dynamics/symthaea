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
