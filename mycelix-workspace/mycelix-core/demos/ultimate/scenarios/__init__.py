# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Ultimate Demo Scenarios

Four demonstration scenarios showcasing the best FL system ever created:
1. Full Stack - Complete system demo
2. Byzantine Resistance - 45% Byzantine adversaries
3. Privacy Showcase - Differential privacy features
4. Scale Test - Scalability demonstration
"""

from .full_stack import FullStackScenario
from .byzantine_resistance import ByzantineResistanceScenario
from .privacy_showcase import PrivacyShowcaseScenario
from .scale_test import ScaleTestScenario

__all__ = [
    "FullStackScenario",
    "ByzantineResistanceScenario",
    "PrivacyShowcaseScenario",
    "ScaleTestScenario",
]
