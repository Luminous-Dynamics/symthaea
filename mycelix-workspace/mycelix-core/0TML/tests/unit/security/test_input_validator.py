# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Tests for input validation module."""

import pytest
import math
from zerotrustml.security.input_validator import (
    GradientValidator,
    GradientValidationError,
    GradientConstraints,
    DIDValidator,
    DIDValidationError,
    DIDConstraints,
    ContractCallValidator,
    ContractValidationError,
    InputValidator,
    ValidationError,
)


class TestGradientValidator:
    """Tests for gradient validation."""
    
    def setup_method(self):
        self.validator = GradientValidator()
    
    def test_valid_gradient(self):
        """Valid gradient should pass."""
        gradient = [0.1, -0.2, 0.3, 0.0, -0.5]
        result = self.validator.validate(gradient)
        assert result == gradient
    
    def test_numpy_array(self):
        """Should handle numpy arrays."""
        pytest.importorskip("numpy")
        import numpy as np
        gradient = np.array([0.1, 0.2, 0.3])
        result = self.validator.validate(gradient)
        assert len(result) == 3
    
    def test_none_gradient(self):
        """None gradient should fail."""
        with pytest.raises(GradientValidationError, match="cannot be None"):
            self.validator.validate(None)
    
    def test_empty_gradient(self):
        """Empty gradient should fail (below min dimension)."""
        with pytest.raises(GradientValidationError, match="below minimum"):
            self.validator.validate([])
    
    def test_exceeds_max_dimension(self):
        """Gradient exceeding max dimension should fail."""
        validator = GradientValidator(GradientConstraints(max_dimension=10))
        with pytest.raises(GradientValidationError, match="exceeds maximum"):
            validator.validate([0.1] * 100)
    
    def test_nan_rejected(self):
        """NaN values should be rejected by default."""
        gradient = [0.1, float('nan'), 0.3]
        with pytest.raises(GradientValidationError, match="NaN"):
            self.validator.validate(gradient)
    
    def test_nan_allowed(self):
        """NaN values should be allowed when configured."""
        validator = GradientValidator(GradientConstraints(allow_nan=True))
        gradient = [0.1, float('nan'), 0.3]
        result = validator.validate(gradient)
        assert result[1] == 0.0  # NaN replaced with 0
    
    def test_inf_rejected(self):
        """Infinite values should be rejected by default."""
        gradient = [0.1, float('inf'), 0.3]
        with pytest.raises(GradientValidationError, match="Infinite"):
            self.validator.validate(gradient)
    
    def test_value_too_large(self):
        """Values exceeding max should fail."""
        validator = GradientValidator(GradientConstraints(max_value=1.0))
        gradient = [0.1, 2.0, 0.3]
        with pytest.raises(GradientValidationError, match="exceeds max"):
            validator.validate(gradient)
    
    def test_value_too_small(self):
        """Values below min should fail."""
        validator = GradientValidator(GradientConstraints(min_value=-1.0))
        gradient = [0.1, -2.0, 0.3]
        with pytest.raises(GradientValidationError, match="below min"):
            validator.validate(gradient)
    
    def test_l2_norm_exceeded(self):
        """L2 norm exceeding max should fail."""
        validator = GradientValidator(GradientConstraints(max_l2_norm=1.0))
        gradient = [10.0, 10.0, 10.0]  # L2 norm ≈ 17.3
        with pytest.raises(GradientValidationError, match="L2 norm"):
            validator.validate(gradient)
    
    def test_non_numeric_value(self):
        """Non-numeric values should fail."""
        gradient = [0.1, "not a number", 0.3]
        with pytest.raises(GradientValidationError, match="Non-numeric"):
            self.validator.validate(gradient)
    
    def test_clip_to_norm(self):
        """Gradient clipping should work correctly."""
        gradient = [3.0, 4.0]  # L2 norm = 5
        result = self.validator.clip_to_norm(gradient, 2.5)
        # Should be scaled to norm 2.5
        new_norm = math.sqrt(sum(v*v for v in result))
        assert abs(new_norm - 2.5) < 1e-10
    
    def test_validate_batch(self):
        """Batch validation should work."""
        gradients = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        results = self.validator.validate_batch(gradients)
        assert len(results) == 3


class TestDIDValidator:
    """Tests for DID validation."""
    
    def setup_method(self):
        self.validator = DIDValidator()
    
    def test_valid_mycelix_did(self):
        """Valid Mycelix DID should pass."""
        did = "did:mycelix:abc123"
        result = self.validator.validate(did)
        assert result == did
    
    def test_valid_key_did(self):
        """Valid key DID should pass."""
        did = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
        result = self.validator.validate(did)
        assert result == did
    
    def test_invalid_format(self):
        """Invalid DID format should fail."""
        with pytest.raises(DIDValidationError, match="Invalid DID format"):
            self.validator.validate("not-a-did")
    
    def test_invalid_method(self):
        """Unknown DID method should fail."""
        with pytest.raises(DIDValidationError, match="method .* not allowed"):
            self.validator.validate("did:unknown:abc123")
    
    def test_non_string(self):
        """Non-string input should fail."""
        with pytest.raises(DIDValidationError, match="must be string"):
            self.validator.validate(12345)
    
    def test_too_long(self):
        """DID exceeding max length should fail."""
        validator = DIDValidator(DIDConstraints(max_length=20))
        with pytest.raises(DIDValidationError, match="exceeds maximum"):
            validator.validate("did:mycelix:" + "a" * 100)
    
    def test_extract_method(self):
        """Method extraction should work."""
        did = "did:mycelix:abc123"
        method = self.validator.extract_method(did)
        assert method == "mycelix"
    
    def test_extract_identifier(self):
        """Identifier extraction should work."""
        did = "did:mycelix:abc123"
        identifier = self.validator.extract_identifier(did)
        assert identifier == "abc123"


class TestContractCallValidator:
    """Tests for contract call validation."""
    
    def setup_method(self):
        self.validator = ContractCallValidator({
            "ReputationAnchor": ["storeReputationRoot", "verifyReputation"],
            "MycelixRegistry": ["registerDID"],
        })
    
    def test_valid_address(self):
        """Valid Ethereum address should pass."""
        address = "0x742d35Cc6634C0532925a3b844Bc9e7595f42012"
        result = self.validator.validate_address(address)
        assert result == address.lower()
    
    def test_invalid_address(self):
        """Invalid Ethereum address should fail."""
        with pytest.raises(ContractValidationError, match="Invalid Ethereum address"):
            self.validator.validate_address("0xinvalid")
    
    def test_valid_bytes32(self):
        """Valid bytes32 should pass."""
        value = "0x" + "a" * 64
        result = self.validator.validate_bytes32(value)
        assert result == value.lower()
    
    def test_invalid_bytes32(self):
        """Invalid bytes32 should fail."""
        with pytest.raises(ContractValidationError, match="Invalid bytes32"):
            self.validator.validate_bytes32("0xshort")
    
    def test_valid_uint256(self):
        """Valid uint256 should pass."""
        result = self.validator.validate_uint256(12345)
        assert result == 12345
    
    def test_negative_uint256(self):
        """Negative uint256 should fail."""
        with pytest.raises(ContractValidationError, match="cannot be negative"):
            self.validator.validate_uint256(-1)
    
    def test_uint256_overflow(self):
        """uint256 overflow should fail."""
        with pytest.raises(ContractValidationError, match="exceeds uint256"):
            self.validator.validate_uint256(2**256)
    
    def test_valid_contract_call(self):
        """Valid contract call should pass."""
        result = self.validator.validate_contract_call(
            "ReputationAnchor",
            "storeReputationRoot",
            {"root": "0x" + "a" * 64}
        )
        assert "root" in result
    
    def test_unknown_contract(self):
        """Unknown contract should fail."""
        with pytest.raises(ContractValidationError, match="Unknown contract"):
            self.validator.validate_contract_call("UnknownContract", "someFunction", {})
    
    def test_unknown_function(self):
        """Unknown function should fail."""
        with pytest.raises(ContractValidationError, match="not allowed"):
            self.validator.validate_contract_call("ReputationAnchor", "unknownFunction", {})


class TestInputValidator:
    """Tests for unified input validator."""
    
    def test_sanitize_string(self):
        """String sanitization should work."""
        result = InputValidator.sanitize_string("Hello\x00World")
        assert result == "HelloWorld"
    
    def test_sanitize_truncates(self):
        """Long strings should be truncated."""
        long_string = "a" * 2000
        result = InputValidator.sanitize_string(long_string, max_length=100)
        assert len(result) == 100
    
    def test_hash_input(self):
        """Input hashing should work."""
        result = InputValidator.hash_input(b"test data")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)
    
    def test_unified_validators(self):
        """Unified validator should have all sub-validators."""
        validator = InputValidator()
        assert hasattr(validator, 'gradient')
        assert hasattr(validator, 'did')
        assert hasattr(validator, 'contract')
