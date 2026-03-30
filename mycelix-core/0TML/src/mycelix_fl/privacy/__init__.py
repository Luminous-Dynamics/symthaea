# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MycelixFL Privacy Module

Differential privacy and secure aggregation for federated learning.

Author: Luminous Dynamics
Date: December 31, 2025
"""

from mycelix_fl.privacy.differential_privacy import (
    DifferentialPrivacy,
    DPConfig,
    GaussianMechanism,
    LaplaceMechanism,
    clip_gradients,
    add_noise,
    compute_privacy_budget,
    PrivacyAccountant,
)

__all__ = [
    "DifferentialPrivacy",
    "DPConfig",
    "GaussianMechanism",
    "LaplaceMechanism",
    "clip_gradients",
    "add_noise",
    "compute_privacy_budget",
    "PrivacyAccountant",
]
