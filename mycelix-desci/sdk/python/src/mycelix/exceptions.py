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
