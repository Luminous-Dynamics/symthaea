# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix-DeSci Exceptions
"""


class MycelixError(Exception):
    """Base exception for Mycelix SDK"""

    pass


class ClaimNotFoundError(MycelixError):
    """Claim not found"""

    pass


class ValidationError(MycelixError):
    """Validation error"""

    pass


class AuthenticationError(MycelixError):
    """Authentication failed"""

    pass


class RateLimitError(MycelixError):
    """Rate limit exceeded"""

    pass


class NetworkError(MycelixError):
    """Network error"""

    pass


class ServerError(MycelixError):
    """Server error"""

    pass
