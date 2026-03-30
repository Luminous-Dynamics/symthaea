# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Core REST API

Provides HTTP endpoints for ecosystem interaction.

Endpoints:
    GET /health - Service health check
    GET /status - Ecosystem status summary
    GET /trust/{agent_id} - Agent trust score lookup
    POST /pogq/validate - Validate a Proof of Gradient Quality
"""

from .server import app, run_server

__all__ = ["app", "run_server"]
