# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix FL Core Module

Contains the unified FL orchestrator and phi measurement.
"""

from mycelix_fl.core.unified_fl import MycelixFL, FLConfig, RoundResult
from mycelix_fl.core.phi_measurement import (
    HypervectorPhiMeasurer,
    PhiMetrics,
    PhiMeasurementResult,
    measure_phi,
)

__all__ = [
    "MycelixFL",
    "FLConfig",
    "RoundResult",
    "HypervectorPhiMeasurer",
    "PhiMetrics",
    "PhiMeasurementResult",
    "measure_phi",
]
