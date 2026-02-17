"""Mycelix-DeSci ML Package

Machine learning and federated learning components for decentralized science.
"""

__version__ = "0.1.0"
__author__ = "Mycelix Contributors"

from .pogq import PoGQValidator
from .fl import FederatedClient, FederatedServer

__all__ = ["PoGQValidator", "FederatedClient", "FederatedServer"]
