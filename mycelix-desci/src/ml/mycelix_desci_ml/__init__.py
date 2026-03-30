# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Mycelix-DeSci ML Package

Machine learning and federated learning components for decentralized science.
"""

__version__ = "0.1.0"
__author__ = "Mycelix Contributors"

from .pogq import PoGQValidator
from .fl import FederatedClient, FederatedServer

__all__ = ["PoGQValidator", "FederatedClient", "FederatedServer"]
