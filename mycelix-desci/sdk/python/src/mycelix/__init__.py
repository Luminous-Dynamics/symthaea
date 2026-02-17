"""
Mycelix-Py: Python SDK for Mycelix-DeSci

Official Python client for the Mycelix-DeSci decentralized science platform.
"""

from .client import MycelixClient, AsyncMycelixClient
from .models import (
    Claim,
    ClaimContent,
    EpistemicTier,
    Provenance,
    Verification,
    TrustScore,
    QueryResult,
)
from .exceptions import (
    MycelixError,
    ClaimNotFoundError,
    ValidationError,
    AuthenticationError,
    RateLimitError,
)

__version__ = "0.1.0"
__all__ = [
    # Clients
    "MycelixClient",
    "AsyncMycelixClient",
    # Models
    "Claim",
    "ClaimContent",
    "EpistemicTier",
    "Provenance",
    "Verification",
    "TrustScore",
    "QueryResult",
    # Exceptions
    "MycelixError",
    "ClaimNotFoundError",
    "ValidationError",
    "AuthenticationError",
    "RateLimitError",
]
