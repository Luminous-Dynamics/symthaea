"""Federated Learning

Client and server implementations for federated learning workflows.
"""

from .client import FederatedClient
from .server import FederatedServer

__all__ = ["FederatedClient", "FederatedServer"]
