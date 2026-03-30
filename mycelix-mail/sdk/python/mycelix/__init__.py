# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Mail SDK for Python

Official SDK for integrating with Mycelix Mail API.

Example:
    >>> from mycelix import MycelixClient
    >>> client = MycelixClient(api_key="your-api-key")
    >>> emails = client.emails.list(folder="inbox")
"""

from .client import MycelixClient
from .models import (
    Email,
    Contact,
    TrustAttestation,
    TrustScore,
    Folder,
    Label,
    WebhookConfig,
)
from .exceptions import MycelixError, AuthenticationError, RateLimitError, NotFoundError

__version__ = "1.0.0"
__all__ = [
    "MycelixClient",
    "Email",
    "Contact",
    "TrustAttestation",
    "TrustScore",
    "Folder",
    "Label",
    "WebhookConfig",
    "MycelixError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
]
