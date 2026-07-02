# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
