# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Proof of Gradient Quality (PoGQ) implementation in Python

Byzantine-resistant gradient validation for federated learning.
"""

from .validator import PoGQValidator
from .aggregator import GradientAggregator
from .detector import ByzantineDetector

__all__ = ["PoGQValidator", "GradientAggregator", "ByzantineDetector"]
