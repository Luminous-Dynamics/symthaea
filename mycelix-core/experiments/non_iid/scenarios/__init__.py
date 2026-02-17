"""
Non-IID Scenario Modules for Mycelix FL Validation

This package provides scenarios for testing Byzantine detection
under various Non-IID data distributions:

- label_skew: Dirichlet-based label distribution heterogeneity
- feature_skew: Domain shift and feature transformation
- quantity_skew: Data quantity imbalance across nodes
- combined_skew: All skew types combined
"""

from .label_skew import LabelSkewScenario, LabelSkewConfig
from .feature_skew import FeatureSkewScenario, FeatureSkewConfig
from .quantity_skew import QuantitySkewScenario, QuantitySkewConfig
from .combined_skew import CombinedSkewScenario, CombinedSkewConfig

__all__ = [
    "LabelSkewScenario",
    "LabelSkewConfig",
    "FeatureSkewScenario",
    "FeatureSkewConfig",
    "QuantitySkewScenario",
    "QuantitySkewConfig",
    "CombinedSkewScenario",
    "CombinedSkewConfig",
]
