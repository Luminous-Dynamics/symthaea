# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Calibration Blob: Tamper-Evident Detector Parameters
====================================================

Serializes and hashes all detector calibration parameters:
- PCA components and variance
- Conformal quantiles and alpha
- Mondrian profiles
- EMA beta and hysteresis params

Enables:
- Tamper detection via SHA-256 hash
- Reproducibility via parameter versioning
- Audit compliance

Usage:
    from src.provenance import CalibrationBlob, save_calibration

    # Create blob
    blob = CalibrationBlob(
        detector_name="pogq_v4.1",
        pca_components=components,
        conformal_quantile=q_alpha,
        mondrian_profiles=["label", "writer_size"],
        alpha=0.10,
        ema_beta=0.85,
        warmup_rounds=3,
        hysteresis_k=2,
        hysteresis_m=3
    )

    # Save with hash
    save_calibration(blob, "calibrations/pogq_v4_1_emnist.json")

    # Verify later
    loaded = load_calibration("calibrations/pogq_v4_1_emnist.json")
    assert loaded.verify_hash(), "Calibration tampered!"

Author: Luminous Dynamics
Date: November 8, 2025
Status: Scaffolding (Phase 4 preparation)
"""

import json
import hashlib
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class CalibrationBlob:
    """
    Tamper-evident container for detector calibration parameters.

    All numeric data is serialized deterministically for hashing.
    """

    # Metadata
    detector_name: str
    version: str = "1.0"
    timestamp: Optional[str] = None

    # PCA parameters
    pca_components: Optional[np.ndarray] = None  # Shape: (n_components, n_features)
    pca_mean: Optional[np.ndarray] = None
    pca_variance: Optional[np.ndarray] = None
    n_components: Optional[int] = None

    # Conformal parameters
    conformal_quantile: Optional[float] = None
    alpha: float = 0.10
    margin: float = 0.02

    # Mondrian parameters
    mondrian_profiles: List[str] = field(default_factory=list)
    bucket_quantiles: Optional[Dict[str, List[float]]] = None

    # EMA parameters
    ema_beta: Optional[float] = None
    warmup_rounds: Optional[int] = None

    # Hysteresis parameters
    hysteresis_k: Optional[int] = None
    hysteresis_m: Optional[int] = None

    # Hash (computed on save, verified on load)
    sha256_hash: Optional[str] = None

    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            from datetime import datetime
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of all parameters (excluding hash itself).

        Uses deterministic JSON serialization for reproducibility.
        """
        # Create copy without hash field
        data = {k: v for k, v in self.to_dict().items() if k != 'sha256_hash'}

        # Serialize deterministically
        serialized = self._deterministic_serialize(data)

        # Hash
        hash_obj = hashlib.sha256(serialized.encode('utf-8'))
        return hash_obj.hexdigest()

    def _deterministic_serialize(self, obj: Any) -> str:
        """
        Deterministic JSON serialization for hashing.

        Handles NumPy arrays by converting to nested lists.
        """
        def convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.integer):
                return int(o)
            return o

        # Sort keys for determinism
        return json.dumps(obj, sort_keys=True, default=convert)

    def verify_hash(self) -> bool:
        """
        Verify that stored hash matches recomputed hash.

        Returns:
            True if hash is valid, False if tampered
        """
        if self.sha256_hash is None:
            return False

        recomputed = self.compute_hash()
        return recomputed == self.sha256_hash

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)

        # Convert NumPy arrays to lists
        if self.pca_components is not None:
            data['pca_components'] = self.pca_components.tolist()
        if self.pca_mean is not None:
            data['pca_mean'] = self.pca_mean.tolist()
        if self.pca_variance is not None:
            data['pca_variance'] = self.pca_variance.tolist()

        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationBlob':
        """Load from dictionary."""
        # Convert lists back to NumPy arrays
        if data.get('pca_components') is not None:
            data['pca_components'] = np.array(data['pca_components'])
        if data.get('pca_mean') is not None:
            data['pca_mean'] = np.array(data['pca_mean'])
        if data.get('pca_variance') is not None:
            data['pca_variance'] = np.array(data['pca_variance'])

        return cls(**data)


def save_calibration(blob: CalibrationBlob, path: str):
    """
    Save calibration blob with hash to JSON file.

    Args:
        blob: CalibrationBlob instance
        path: Output file path
    """
    # Compute hash before saving
    blob.sha256_hash = blob.compute_hash()

    # Save to JSON
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(blob.to_dict(), f, indent=2)

    print(f"✅ Calibration saved: {path}")
    print(f"   SHA-256: {blob.sha256_hash[:16]}...")


def load_calibration(path: str, verify: bool = True) -> CalibrationBlob:
    """
    Load calibration blob from JSON file.

    Args:
        path: Input file path
        verify: If True, verify hash after loading

    Returns:
        CalibrationBlob instance

    Raises:
        ValueError: If verification fails
    """
    with open(path, 'r') as f:
        data = json.load(f)

    blob = CalibrationBlob.from_dict(data)

    if verify:
        if not blob.verify_hash():
            raise ValueError(f"Calibration hash verification failed: {path}")
        print(f"✅ Calibration verified: {path}")

    return blob


# ==================================================================
# Testing
# ==================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Calibration Blob: Unit Tests")
    print("=" * 70)

    # Test 1: Create and save
    print("\nTest 1: Create and save calibration blob")

    blob = CalibrationBlob(
        detector_name="pogq_v4.1",
        pca_components=np.random.randn(10, 100),
        pca_mean=np.random.randn(100),
        pca_variance=np.random.rand(10),
        n_components=10,
        conformal_quantile=0.95,
        alpha=0.10,
        mondrian_profiles=["label", "writer_size"],
        ema_beta=0.85,
        warmup_rounds=3,
        hysteresis_k=2,
        hysteresis_m=3
    )

    # Save
    test_path = "/tmp/test_calibration.json"
    save_calibration(blob, test_path)
    print(f"✅ Saved to {test_path}")

    # Test 2: Load and verify
    print("\nTest 2: Load and verify hash")
    loaded = load_calibration(test_path, verify=True)
    assert loaded.sha256_hash == blob.sha256_hash
    print("✅ Hash verified successfully")

    # Test 3: Tamper detection
    print("\nTest 3: Tamper detection")
    # Modify file
    with open(test_path, 'r') as f:
        data = json.load(f)
    data['alpha'] = 0.20  # Tamper!
    with open(test_path, 'w') as f:
        json.dump(data, f)

    # Try to load
    try:
        load_calibration(test_path, verify=True)
        print("❌ Tamper detection failed!")
    except ValueError as e:
        print(f"✅ Tamper detected: {e}")

    print("\n" + "=" * 70)
    print("All tests passed! Calibration blob ready for Phase 4.")
    print("=" * 70)
