# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Exceptions for Mycelix Mail SDK
"""


class MycelixError(Exception):
    """Base exception for Mycelix API errors."""

    def __init__(self, code: str, message: str, status: int = 0):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(f"[{code}] {message}")


class AuthenticationError(MycelixError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__("auth_error", message, 401)


class RateLimitError(MycelixError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__("rate_limit", message, 429)


class NotFoundError(MycelixError):
    """Raised when a resource is not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__("not_found", message, 404)


class ValidationError(MycelixError):
    """Raised when request validation fails."""

    def __init__(self, message: str, errors: dict = None):
        self.errors = errors or {}
        super().__init__("validation_error", message, 400)


class PermissionError(MycelixError):
    """Raised when user lacks permission."""

    def __init__(self, message: str = "Permission denied"):
        super().__init__("permission_denied", message, 403)


class ConflictError(MycelixError):
    """Raised when there's a conflict (e.g., duplicate)."""

    def __init__(self, message: str = "Resource conflict"):
        super().__init__("conflict", message, 409)


class ServerError(MycelixError):
    """Raised when the server encounters an error."""

    def __init__(self, message: str = "Internal server error"):
        super().__init__("server_error", message, 500)


class ConnectionError(MycelixError):
    """Raised when connection to API fails."""

    def __init__(self, message: str = "Failed to connect to API"):
        super().__init__("connection_error", message, 0)


class TimeoutError(MycelixError):
    """Raised when request times out."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__("timeout", message, 0)
