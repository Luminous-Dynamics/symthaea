# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix CLI - Unified Command-Line Interface for Mycelix Federated Learning

A beautiful, guided experience for using the Byzantine-resilient FL system.

Features:
- Interactive demos with multiple scenarios
- Project initialization and configuration
- Benchmark harness for performance testing
- Validation and health checks
- Real-time progress with rich terminal output

Author: Luminous Dynamics
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Luminous Dynamics"

from .cli import app

__all__ = ["app", "__version__"]
