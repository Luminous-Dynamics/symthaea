# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Error Categories, Severity Levels, and Error Codes

Provides standardized classification for all Mycelix errors:
- Categories group related errors (Network, Storage, etc.)
- Severity levels indicate impact (DEBUG to CRITICAL)
- Error codes provide unique identifiers for programmatic handling
"""

from enum import Enum, auto


class ErrorCategory(Enum):
    """
    Standard error categories for Mycelix ecosystem.

    Each category maps to a distinct failure domain.
    """
    # Infrastructure
    NETWORK = "network"          # Connection, timeout, protocol
    STORAGE = "storage"          # Database, filesystem, DHT
    CRYPTO = "crypto"            # Encryption, signing, verification

    # Application
    VALIDATION = "validation"    # Input validation, schema
    ZOME = "zome"                # Holochain zome call failures
    GOVERNANCE = "governance"    # Authorization, capabilities
    IDENTITY = "identity"        # DID, credentials, verification

    # Security
    BYZANTINE = "byzantine"      # Detected attacks, malicious behavior
    AGGREGATION = "aggregation"  # FL aggregation failures

    # System
    CONFIGURATION = "config"     # Config parsing, missing values
    INTERNAL = "internal"        # Unexpected internal errors
    TIMEOUT = "timeout"          # Operation timeouts


class ErrorSeverity(Enum):
    """
    Error severity levels.

    Maps to standard logging levels for consistency:
    - DEBUG: Diagnostic information, not user-facing
    - INFO: Informational, expected error conditions
    - WARNING: Unexpected but recoverable
    - ERROR: Significant problem requiring attention
    - CRITICAL: System failure, immediate action required
    """
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def log_level(self) -> int:
        """Return Python logging level."""
        import logging
        return {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }[self]

    def __lt__(self, other):
        order = [
            ErrorSeverity.DEBUG,
            ErrorSeverity.INFO,
            ErrorSeverity.WARNING,
            ErrorSeverity.ERROR,
            ErrorSeverity.CRITICAL,
        ]
        return order.index(self) < order.index(other)


class ErrorCode(Enum):
    """
    Standardized error codes for Mycelix ecosystem.

    Format: CATEGORY_SPECIFIC_ERROR
    Code ranges:
    - 1xxx: General/Internal
    - 2xxx: Configuration
    - 3xxx: Network/Connection
    - 4xxx: Byzantine/Security
    - 5xxx: Proof/Verification
    - 6xxx: Identity
    - 7xxx: Governance
    - 8xxx: Storage
    - 9xxx: Aggregation
    - 10xxx: Crypto/Encryption
    """
    # General errors (1xxx)
    UNKNOWN = "ERR_1000"
    INTERNAL = "ERR_1001"
    TIMEOUT = "ERR_1002"
    VALIDATION = "ERR_1003"
    NOT_IMPLEMENTED = "ERR_1004"
    INVALID_STATE = "ERR_1005"

    # Configuration errors (2xxx)
    CONFIG_INVALID = "ERR_2000"
    CONFIG_MISSING = "ERR_2001"
    CONFIG_TYPE_ERROR = "ERR_2002"
    CONFIG_CONSTRAINT = "ERR_2003"
    CONFIG_FILE_NOT_FOUND = "ERR_2004"

    # Network errors (3xxx)
    CONNECTION_FAILED = "ERR_3000"
    CONNECTION_TIMEOUT = "ERR_3001"
    CONNECTION_REFUSED = "ERR_3002"
    CONNECTION_LOST = "ERR_3003"
    PROTOCOL_ERROR = "ERR_3004"
    DNS_RESOLUTION_FAILED = "ERR_3005"

    # Holochain-specific network (31xx)
    HOLOCHAIN_CONNECTION_FAILED = "ERR_3100"
    HOLOCHAIN_CONDUCTOR_UNAVAILABLE = "ERR_3101"
    HOLOCHAIN_ZOME_CALL_FAILED = "ERR_3102"
    HOLOCHAIN_CELL_NOT_FOUND = "ERR_3103"
    HOLOCHAIN_APP_NOT_INSTALLED = "ERR_3104"

    # Postgres-specific (32xx)
    POSTGRES_CONNECTION_FAILED = "ERR_3200"
    POSTGRES_QUERY_FAILED = "ERR_3201"

    # WebSocket (33xx)
    WEBSOCKET_ERROR = "ERR_3300"
    WEBSOCKET_CLOSED = "ERR_3301"

    # Byzantine detection errors (4xxx)
    BYZANTINE_DETECTED = "ERR_4000"
    BYZANTINE_GRADIENT_POISONING = "ERR_4001"
    BYZANTINE_MODEL_CORRUPTION = "ERR_4002"
    BYZANTINE_SYBIL_ATTACK = "ERR_4003"
    BYZANTINE_FREE_RIDER = "ERR_4004"
    BYZANTINE_COLLUSION = "ERR_4005"
    POGQ_BELOW_THRESHOLD = "ERR_4100"
    REPUTATION_TOO_LOW = "ERR_4101"

    # Proof verification errors (5xxx)
    PROOF_VERIFICATION_FAILED = "ERR_5000"
    PROOF_INVALID_FORMAT = "ERR_5001"
    PROOF_SIGNATURE_INVALID = "ERR_5002"
    PROOF_EXPIRED = "ERR_5003"
    ZKPOC_VERIFICATION_FAILED = "ERR_5100"
    ZKPOC_RANGE_PROOF_INVALID = "ERR_5101"
    ZKPOC_COMMITMENT_MISMATCH = "ERR_5102"

    # Identity errors (6xxx)
    IDENTITY_ERROR = "ERR_6000"
    IDENTITY_NOT_FOUND = "ERR_6001"
    IDENTITY_VERIFICATION_FAILED = "ERR_6002"
    IDENTITY_EXPIRED = "ERR_6003"
    DID_RESOLUTION_FAILED = "ERR_6100"
    DID_INVALID_FORMAT = "ERR_6101"
    CREDENTIAL_INVALID = "ERR_6200"
    CREDENTIAL_EXPIRED = "ERR_6201"
    CREDENTIAL_REVOKED = "ERR_6202"
    ASSURANCE_LEVEL_INSUFFICIENT = "ERR_6300"

    # Governance errors (7xxx)
    GOVERNANCE_ERROR = "ERR_7000"
    GOVERNANCE_UNAUTHORIZED = "ERR_7001"
    GOVERNANCE_PROPOSAL_NOT_FOUND = "ERR_7002"
    GOVERNANCE_VOTE_FAILED = "ERR_7003"
    GOVERNANCE_QUORUM_NOT_MET = "ERR_7004"
    GOVERNANCE_VOTING_CLOSED = "ERR_7005"
    CAPABILITY_DENIED = "ERR_7100"
    CAPABILITY_NOT_FOUND = "ERR_7101"
    CAPABILITY_RATE_LIMITED = "ERR_7102"
    GUARDIAN_AUTHORIZATION_REQUIRED = "ERR_7200"
    GUARDIAN_AUTHORIZATION_EXPIRED = "ERR_7201"

    # Storage errors (8xxx)
    STORAGE_ERROR = "ERR_8000"
    STORAGE_READ_FAILED = "ERR_8001"
    STORAGE_WRITE_FAILED = "ERR_8002"
    STORAGE_NOT_FOUND = "ERR_8003"
    STORAGE_INTEGRITY_ERROR = "ERR_8004"
    STORAGE_FULL = "ERR_8005"
    GRADIENT_NOT_FOUND = "ERR_8100"
    GRADIENT_CORRUPTED = "ERR_8101"
    CLAIM_NOT_FOUND = "ERR_8102"

    # Aggregation errors (9xxx)
    AGGREGATION_ERROR = "ERR_9000"
    AGGREGATION_INSUFFICIENT_GRADIENTS = "ERR_9001"
    AGGREGATION_ALGORITHM_FAILED = "ERR_9002"
    AGGREGATION_QUORUM_NOT_MET = "ERR_9003"
    AGGREGATION_ROUND_EXPIRED = "ERR_9004"

    # Encryption errors (10xxx)
    ENCRYPTION_ERROR = "ERR_10000"
    ENCRYPTION_KEY_MISSING = "ERR_10001"
    ENCRYPTION_KEY_INVALID = "ERR_10002"
    DECRYPTION_FAILED = "ERR_10003"
    SIGNATURE_INVALID = "ERR_10004"
    HASH_MISMATCH = "ERR_10005"

    @property
    def category(self) -> ErrorCategory:
        """Return the error category based on code range."""
        code_num = int(self.value.split("_")[1])
        if 1000 <= code_num < 2000:
            return ErrorCategory.INTERNAL
        elif 2000 <= code_num < 3000:
            return ErrorCategory.CONFIGURATION
        elif 3000 <= code_num < 4000:
            return ErrorCategory.NETWORK
        elif 4000 <= code_num < 5000:
            return ErrorCategory.BYZANTINE
        elif 5000 <= code_num < 6000:
            return ErrorCategory.CRYPTO
        elif 6000 <= code_num < 7000:
            return ErrorCategory.IDENTITY
        elif 7000 <= code_num < 8000:
            return ErrorCategory.GOVERNANCE
        elif 8000 <= code_num < 9000:
            return ErrorCategory.STORAGE
        elif 9000 <= code_num < 10000:
            return ErrorCategory.AGGREGATION
        else:
            return ErrorCategory.CRYPTO
