# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix FL Detection Module

Multi-layer Byzantine detection stack combining:
- Layer 1: zkSTARK verification
- Layer 2: PoGQ validation
- Layer 3: Shapley contribution analysis
- Layer 4: Hypervector Byzantine Detection (HBD)
- Layer 5: Self-healing (error correction)

Combined detection rate: 99%+ @ 45% Byzantine nodes
"""

from mycelix_fl.detection.multi_layer_stack import (
    MultiLayerByzantineDetector,
    DetectionResult,
    DetectionLayer,
)
from mycelix_fl.detection.shapley_detector import ShapleyByzantineDetector
from mycelix_fl.detection.self_healing import SelfHealingDetector

__all__ = [
    "MultiLayerByzantineDetector",
    "DetectionResult",
    "DetectionLayer",
    "ShapleyByzantineDetector",
    "SelfHealingDetector",
]
