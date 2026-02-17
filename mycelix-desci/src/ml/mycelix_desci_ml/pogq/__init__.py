"""Proof of Gradient Quality (PoGQ) implementation in Python

Byzantine-resistant gradient validation for federated learning.
"""

from .validator import PoGQValidator
from .aggregator import GradientAggregator
from .detector import ByzantineDetector

__all__ = ["PoGQValidator", "GradientAggregator", "ByzantineDetector"]
