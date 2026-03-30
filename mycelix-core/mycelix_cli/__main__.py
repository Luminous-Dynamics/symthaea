#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Entry point for `python -m mycelix_cli`

This allows the CLI to be run as a module:
    python -m mycelix_cli --help
    python -m mycelix_cli demo --scenario healthcare
"""

from .cli import app

if __name__ == "__main__":
    app()
