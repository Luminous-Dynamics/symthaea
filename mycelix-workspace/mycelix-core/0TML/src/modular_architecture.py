# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Compatibility shim for legacy imports.

Phase 10 relocated the implementation to `zerotrustml.modular_architecture`.
Older modules/tests still import `modular_architecture` at the top level, so this
module re-exports the new package to preserve backwards compatibility.
"""

from zerotrustml.modular_architecture import *  # noqa: F401,F403
