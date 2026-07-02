# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix CLI Commands

This package contains all subcommands for the Mycelix CLI:
- demo: Run interactive FL demonstrations
- init: Initialize new Mycelix projects
- benchmark: Run performance benchmarks
- validate: Validate system and run tests
- status: Show system status
"""

from . import demo, init, benchmark, validate, status

__all__ = ["demo", "init", "benchmark", "validate", "status"]
