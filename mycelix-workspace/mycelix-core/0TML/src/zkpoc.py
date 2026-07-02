# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Compatibility shim: expose experimental ZKPoC module at legacy import path.
"""

from zerotrustml.experimental.zkpoc import *  # noqa: F401,F403
