# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain bridge modules for MATL and FL pipeline operations.

Provides typed Python access to MATL-specific and pipeline zome functions
not covered by the general FLCoordinatorClient.

Modules:
- matl: PoGQ v4.1 enhanced evaluation, composite trust scoring, round MATL stats
- fl_pipeline: Validator pipeline, commit/reveal, hypervector similarity
"""

from .matl import MATLClient, create_matl_client
from .fl_pipeline import FLPipelineClient, create_fl_pipeline_client

__all__ = [
    "MATLClient",
    "create_matl_client",
    "FLPipelineClient",
    "create_fl_pipeline_client",
]
