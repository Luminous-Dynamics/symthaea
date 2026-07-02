# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Input Validation for ZeroTrustML Federated Learning.

Provides comprehensive validation for all external inputs to prevent:
- Gradient poisoning attacks
- DID injection attacks
- Contract call manipulation
- Buffer overflow attempts
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import re
import hashlib
import math
from enum import Enum


class ValidationError(Exception):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(f"{field}: {message}" if field else message)


class GradientValidationError(ValidationError):
    """Raised when gradient validation fails."""
    pass


class DIDValidationError(ValidationError):
    """Raised when DID validation fails."""
    pass


class ContractValidationError(ValidationError):
    """Raised when contract call validation fails."""
    pass


@dataclass
class GradientConstraints:
    """Constraints for gradient validation."""
    max_dimension: int = 100_000_000  # 100M parameters max
    min_dimension: int = 1
    max_value: float = 1e6  # Prevent extreme gradients
    min_value: float = -1e6
    max_l2_norm: float = 1e8
    allow_nan: bool = False
    allow_inf: bool = False


@dataclass  
class DIDConstraints:
    """Constraints for DID validation."""
    max_length: int = 256
    allowed_methods: tuple = ("mycelix", "key", "web", "ethr")
    allowed_chars: str = r"[a-zA-Z0-9._\-:]+"


class GradientValidator:
    """Validates federated learning gradient inputs."""
    
    def __init__(self, constraints: GradientConstraints = None):
        self.constraints = constraints or GradientConstraints()
    
    def validate(self, gradient: Any, node_id: str = None) -> List[float]:
        """
        Validate a gradient submission.
        
        Args:
            gradient: The gradient values (list, tuple, or array-like)
            node_id: Optional node identifier for logging
            
        Returns:
            Validated gradient as list of floats
            
        Raises:
            GradientValidationError: If validation fails
        """
        # Type check
        if gradient is None:
            raise GradientValidationError("Gradient cannot be None", "gradient")
        
        # Convert to list
        try:
            if hasattr(gradient, 'tolist'):  # numpy array
                gradient_list = gradient.tolist()
            elif hasattr(gradient, '__iter__'):
                gradient_list = list(gradient)
            else:
                raise GradientValidationError(
                    "Gradient must be iterable", "gradient", type(gradient).__name__
                )
        except Exception as e:
            raise GradientValidationError(
                f"Failed to convert gradient: {e}", "gradient"
            )
        
        # Dimension check
        dim = len(gradient_list)
        if dim < self.constraints.min_dimension:
            raise GradientValidationError(
                f"Gradient dimension {dim} below minimum {self.constraints.min_dimension}",
                "dimension", dim
            )
        if dim > self.constraints.max_dimension:
            raise GradientValidationError(
                f"Gradient dimension {dim} exceeds maximum {self.constraints.max_dimension}",
                "dimension", dim
            )
        
        # Value validation
        validated = []
        l2_norm_sq = 0.0
        
        for i, val in enumerate(gradient_list):
            # Type coercion
            try:
                float_val = float(val)
            except (TypeError, ValueError) as e:
                raise GradientValidationError(
                    f"Non-numeric value at index {i}: {val}",
                    f"gradient[{i}]", val
                )
            
            # NaN check
            if math.isnan(float_val):
                if not self.constraints.allow_nan:
                    raise GradientValidationError(
                        f"NaN value at index {i}",
                        f"gradient[{i}]", float_val
                    )
                float_val = 0.0  # Replace NaN with 0
            
            # Inf check
            if math.isinf(float_val):
                if not self.constraints.allow_inf:
                    raise GradientValidationError(
                        f"Infinite value at index {i}",
                        f"gradient[{i}]", float_val
                    )
                # Clip to max/min
                float_val = self.constraints.max_value if float_val > 0 else self.constraints.min_value
            
            # Range check
            if float_val > self.constraints.max_value:
                raise GradientValidationError(
                    f"Value {float_val} at index {i} exceeds max {self.constraints.max_value}",
                    f"gradient[{i}]", float_val
                )
            if float_val < self.constraints.min_value:
                raise GradientValidationError(
                    f"Value {float_val} at index {i} below min {self.constraints.min_value}",
                    f"gradient[{i}]", float_val
                )
            
            validated.append(float_val)
            l2_norm_sq += float_val * float_val
        
        # L2 norm check
        l2_norm = math.sqrt(l2_norm_sq)
        if l2_norm > self.constraints.max_l2_norm:
            raise GradientValidationError(
                f"L2 norm {l2_norm:.2e} exceeds maximum {self.constraints.max_l2_norm:.2e}",
                "l2_norm", l2_norm
            )
        
        return validated
    
    def validate_batch(self, gradients: List[Any]) -> List[List[float]]:
        """Validate multiple gradients."""
        return [self.validate(g) for g in gradients]
    
    def clip_to_norm(self, gradient: List[float], max_norm: float) -> List[float]:
        """Clip gradient to maximum L2 norm."""
        l2_norm = math.sqrt(sum(v * v for v in gradient))
        if l2_norm <= max_norm:
            return gradient
        scale = max_norm / l2_norm
        return [v * scale for v in gradient]


class DIDValidator:
    """Validates Decentralized Identifier (DID) inputs."""
    
    # DID format: did:<method>:<method-specific-id>
    DID_PATTERN = re.compile(r'^did:([a-z0-9]+):([a-zA-Z0-9._\-:]+)$')
    
    def __init__(self, constraints: DIDConstraints = None):
        self.constraints = constraints or DIDConstraints()
    
    def validate(self, did: str) -> str:
        """
        Validate a DID string.
        
        Args:
            did: The DID string to validate
            
        Returns:
            The validated DID string
            
        Raises:
            DIDValidationError: If validation fails
        """
        if not isinstance(did, str):
            raise DIDValidationError(
                f"DID must be string, got {type(did).__name__}",
                "did", did
            )
        
        # Length check
        if len(did) > self.constraints.max_length:
            raise DIDValidationError(
                f"DID length {len(did)} exceeds maximum {self.constraints.max_length}",
                "length", len(did)
            )
        
        # Format check
        match = self.DID_PATTERN.match(did)
        if not match:
            raise DIDValidationError(
                "Invalid DID format. Expected: did:<method>:<id>",
                "format", did
            )
        
        method = match.group(1)
        identifier = match.group(2)
        
        # Method check
        if method not in self.constraints.allowed_methods:
            raise DIDValidationError(
                f"DID method '{method}' not allowed. Allowed: {self.constraints.allowed_methods}",
                "method", method
            )
        
        # Character check for identifier (prevent injection)
        if not re.match(self.constraints.allowed_chars, identifier):
            raise DIDValidationError(
                "DID contains invalid characters",
                "identifier", identifier
            )
        
        return did
    
    def extract_method(self, did: str) -> str:
        """Extract the method from a validated DID."""
        validated = self.validate(did)
        return validated.split(':')[1]
    
    def extract_identifier(self, did: str) -> str:
        """Extract the method-specific identifier from a validated DID."""
        validated = self.validate(did)
        parts = validated.split(':', 2)
        return parts[2] if len(parts) > 2 else ""


class ContractCallValidator:
    """Validates smart contract call parameters."""
    
    # Ethereum address pattern
    ADDRESS_PATTERN = re.compile(r'^0x[a-fA-F0-9]{40}$')
    
    # Bytes32 pattern
    BYTES32_PATTERN = re.compile(r'^0x[a-fA-F0-9]{64}$')
    
    def __init__(self, known_contracts: Dict[str, List[str]] = None):
        """
        Args:
            known_contracts: Dict mapping contract names to allowed function names
        """
        self.known_contracts = known_contracts or {}
    
    def validate_address(self, address: str, field: str = "address") -> str:
        """Validate an Ethereum address."""
        if not isinstance(address, str):
            raise ContractValidationError(
                f"Address must be string, got {type(address).__name__}",
                field, address
            )
        
        if not self.ADDRESS_PATTERN.match(address):
            raise ContractValidationError(
                "Invalid Ethereum address format",
                field, address
            )
        
        return address.lower()  # Normalize to lowercase
    
    def validate_bytes32(self, value: str, field: str = "bytes32") -> str:
        """Validate a bytes32 value."""
        if not isinstance(value, str):
            raise ContractValidationError(
                f"Bytes32 must be string, got {type(value).__name__}",
                field, value
            )
        
        if not self.BYTES32_PATTERN.match(value):
            raise ContractValidationError(
                "Invalid bytes32 format",
                field, value
            )
        
        return value.lower()
    
    def validate_uint256(self, value: Any, field: str = "uint256") -> int:
        """Validate a uint256 value."""
        try:
            int_val = int(value)
        except (TypeError, ValueError):
            raise ContractValidationError(
                f"Cannot convert to uint256: {value}",
                field, value
            )
        
        if int_val < 0:
            raise ContractValidationError(
                "uint256 cannot be negative",
                field, int_val
            )
        
        if int_val >= 2**256:
            raise ContractValidationError(
                "Value exceeds uint256 maximum",
                field, int_val
            )
        
        return int_val
    
    def validate_contract_call(
        self, 
        contract_name: str, 
        function_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a complete contract call.
        
        Args:
            contract_name: Name of the contract
            function_name: Function to call
            params: Function parameters
            
        Returns:
            Validated parameters
        """
        # Check contract is known
        if contract_name not in self.known_contracts:
            raise ContractValidationError(
                f"Unknown contract: {contract_name}",
                "contract_name", contract_name
            )
        
        # Check function is allowed
        allowed_functions = self.known_contracts[contract_name]
        if function_name not in allowed_functions:
            raise ContractValidationError(
                f"Function '{function_name}' not allowed on {contract_name}",
                "function_name", function_name
            )
        
        return params


class InputValidator:
    """Unified input validator combining all validation types."""
    
    def __init__(
        self,
        gradient_constraints: GradientConstraints = None,
        did_constraints: DIDConstraints = None,
        known_contracts: Dict[str, List[str]] = None,
    ):
        self.gradient = GradientValidator(gradient_constraints)
        self.did = DIDValidator(did_constraints)
        self.contract = ContractCallValidator(known_contracts)
    
    @staticmethod
    def sanitize_string(s: str, max_length: int = 1000) -> str:
        """Sanitize a string input."""
        if not isinstance(s, str):
            raise ValidationError("Expected string", "type", type(s).__name__)
        
        # Truncate
        s = s[:max_length]
        
        # Remove null bytes
        s = s.replace('\x00', '')
        
        # Remove control characters except newline/tab
        s = ''.join(c for c in s if c == '\n' or c == '\t' or (ord(c) >= 32 and ord(c) < 127) or ord(c) > 127)
        
        return s
    
    @staticmethod
    def hash_input(data: bytes) -> str:
        """Create a hash of input data for logging."""
        return hashlib.sha256(data).hexdigest()[:16]


# Default validators for common use cases
default_gradient_validator = GradientValidator()
default_did_validator = DIDValidator()
default_contract_validator = ContractCallValidator({
    "ReputationAnchor": ["storeReputationRoot", "verifyReputation", "verifyAndRecord"],
    "PaymentRouter": ["routePayment", "withdraw"],
    "MycelixRegistry": ["registerDID", "getDID", "updateMetadata"],
})
