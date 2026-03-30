# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL Distributed Module

Async and distributed federated learning support.

Author: Luminous Dynamics
Date: December 31, 2025
"""

from mycelix_fl.distributed.async_fl import (
    AsyncMycelixFL,
    AsyncFLConfig,
    NodeConnection,
    GradientMessage,
    AggregationMessage,
    FLProtocol,
)
from mycelix_fl.distributed.coordinator import (
    FLCoordinator,
    CoordinatorConfig,
    RoundState,
)

__all__ = [
    # Async FL
    "AsyncMycelixFL",
    "AsyncFLConfig",
    "NodeConnection",
    "GradientMessage",
    "AggregationMessage",
    "FLProtocol",
    # Coordinator
    "FLCoordinator",
    "CoordinatorConfig",
    "RoundState",
]
