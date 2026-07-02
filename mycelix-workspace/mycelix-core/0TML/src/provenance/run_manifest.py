# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Run Manifest: Reproducibility & Audit Trail
===========================================

Creates tamper-evident manifest for each experiment run:
- Git commit hash
- Full configuration (YAML)
- Random seeds used
- Artifact file hashes (SHA-256)
- Optional cryptographic signature (Ed25519)

Enables:
- Exact reproducibility
- Audit compliance
- Tamper detection

Usage:
    from src.provenance import create_manifest, verify_manifest

    # Create manifest after experiment
    manifest = create_manifest(
        config_path="configs/sanity_slice.yaml",
        artifact_dir="results/artifacts_20251108_160000",
        seeds=[42, 1337],
        git_commit="a1b2c3d4",
        sign=True  # Optional Ed25519 signature
    )

    # Save
    manifest.save("results/artifacts_20251108_160000/MANIFEST.json")

    # Verify later
    verify_manifest("results/artifacts_20251108_160000/MANIFEST.json")

Author: Luminous Dynamics
Date: November 8, 2025
Status: Scaffolding (Phase 4 preparation)
"""

import json
import hashlib
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path
import yaml


@dataclass
class RunManifest:
    """
    Tamper-evident manifest for experiment run.
    """

    # Version
    version: str = "1.0"

    # Git provenance
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False

    # Configuration
    config_path: str = ""
    config_hash: Optional[str] = None
    config_content: Optional[Dict] = None

    # Execution
    seeds: List[int] = None
    timestamp: Optional[str] = None
    hostname: Optional[str] = None

    # Artifacts
    artifact_dir: str = ""
    artifact_hashes: Dict[str, str] = None  # filename -> SHA-256

    # Signature (optional)
    signature: Optional[str] = None  # Ed25519 signature
    public_key: Optional[str] = None

    def __post_init__(self):
        """Set defaults"""
        if self.timestamp is None:
            from datetime import datetime
            self.timestamp = datetime.utcnow().isoformat() + "Z"

        if self.seeds is None:
            self.seeds = []

        if self.artifact_hashes is None:
            self.artifact_hashes = {}

    def compute_content_hash(self) -> str:
        """
        Compute SHA-256 hash of manifest content (excluding signature).
        """
        # Create copy without signature fields
        data = {k: v for k, v in asdict(self).items()
                if k not in ('signature', 'public_key')}

        # Deterministic serialization
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def sign(self, private_key_path: str):
        """
        Sign manifest with Ed25519 private key.

        Args:
            private_key_path: Path to Ed25519 private key file

        Note: Requires `cryptography` package (optional dependency)
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            print("⚠️  cryptography package not installed, skipping signature")
            return

        # Load private key
        with open(private_key_path, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )

        # Sign content hash
        content_hash = self.compute_content_hash()
        signature_bytes = private_key.sign(content_hash.encode('utf-8'))

        # Store signature and public key
        self.signature = signature_bytes.hex()
        public_key = private_key.public_key()
        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        self.public_key = public_key_bytes.hex()

    def verify_signature(self) -> bool:
        """
        Verify Ed25519 signature.

        Returns:
            True if signature is valid, False otherwise
        """
        if self.signature is None or self.public_key is None:
            return False

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError:
            print("⚠️  cryptography package not installed, cannot verify")
            return False

        # Load public key
        public_key_bytes = bytes.fromhex(self.public_key)
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)

        # Verify signature
        content_hash = self.compute_content_hash()
        signature_bytes = bytes.fromhex(self.signature)

        try:
            public_key.verify(signature_bytes, content_hash.encode('utf-8'))
            return True
        except:
            return False

    def save(self, path: str):
        """Save manifest to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

        print(f"✅ Manifest saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'RunManifest':
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(**data)


def create_manifest(
    config_path: str,
    artifact_dir: str,
    seeds: List[int],
    git_commit: Optional[str] = None,
    sign: bool = False,
    private_key_path: Optional[str] = None
) -> RunManifest:
    """
    Create run manifest for experiment.

    Args:
        config_path: Path to experiment config YAML
        artifact_dir: Directory containing result artifacts
        seeds: List of random seeds used
        git_commit: Git commit hash (auto-detected if None)
        sign: If True, sign with Ed25519 key
        private_key_path: Path to private key (required if sign=True)

    Returns:
        RunManifest instance
    """
    # Get git info
    if git_commit is None:
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            git_commit = "unknown"

    try:
        git_branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
    except:
        git_branch = "unknown"

    try:
        git_status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        git_dirty = len(git_status) > 0
    except:
        git_dirty = False

    # Load config
    with open(config_path, 'r') as f:
        config_content = yaml.safe_load(f)

    # Hash config
    config_str = yaml.dump(config_content, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()

    # Hash all artifacts
    artifact_hashes = {}
    artifact_path = Path(artifact_dir)

    if artifact_path.exists():
        for file in artifact_path.glob('*.json'):
            with open(file, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            artifact_hashes[file.name] = file_hash

    # Get hostname
    import socket
    hostname = socket.gethostname()

    # Create manifest
    manifest = RunManifest(
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        config_path=config_path,
        config_hash=config_hash,
        config_content=config_content,
        seeds=seeds,
        hostname=hostname,
        artifact_dir=artifact_dir,
        artifact_hashes=artifact_hashes
    )

    # Sign if requested
    if sign:
        if private_key_path is None:
            print("⚠️  Signature requested but no private key provided")
        else:
            manifest.sign(private_key_path)

    return manifest


def verify_manifest(manifest_path: str, verbose: bool = True) -> bool:
    """
    Verify manifest integrity.

    Checks:
    1. All artifact files exist
    2. All artifact hashes match
    3. Signature is valid (if present)

    Args:
        manifest_path: Path to MANIFEST.json
        verbose: Print verification details

    Returns:
        True if all checks pass, False otherwise
    """
    manifest = RunManifest.load(manifest_path)

    all_passed = True

    if verbose:
        print("=" * 70)
        print(f"Verifying manifest: {manifest_path}")
        print("=" * 70)

    # Check artifact hashes
    artifact_dir = Path(manifest.artifact_dir)

    for filename, expected_hash in manifest.artifact_hashes.items():
        file_path = artifact_dir / filename

        if not file_path.exists():
            if verbose:
                print(f"❌ Missing: {filename}")
            all_passed = False
            continue

        # Recompute hash
        with open(file_path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        if actual_hash != expected_hash:
            if verbose:
                print(f"❌ Hash mismatch: {filename}")
                print(f"   Expected: {expected_hash[:16]}...")
                print(f"   Actual:   {actual_hash[:16]}...")
            all_passed = False
        elif verbose:
            print(f"✅ Verified: {filename}")

    # Verify signature
    if manifest.signature is not None:
        if manifest.verify_signature():
            if verbose:
                print(f"✅ Signature verified")
        else:
            if verbose:
                print(f"❌ Invalid signature")
            all_passed = False

    if verbose:
        print("=" * 70)
        if all_passed:
            print("✅ Manifest verification PASSED")
        else:
            print("❌ Manifest verification FAILED")
        print("=" * 70)

    return all_passed


# ==================================================================
# Testing
# ==================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Run Manifest: Unit Tests")
    print("=" * 70)

    # Test 1: Create manifest
    print("\nTest 1: Create manifest (without signature)")

    # Create dummy config
    test_config = "/tmp/test_config.yaml"
    with open(test_config, 'w') as f:
        yaml.dump({'dataset': 'mnist', 'attack': 'sign_flip'}, f)

    # Create dummy artifacts
    test_artifact_dir = "/tmp/test_artifacts"
    Path(test_artifact_dir).mkdir(exist_ok=True)

    with open(f"{test_artifact_dir}/detection_metrics.json", 'w') as f:
        json.dump({'auroc': 0.95}, f)

    manifest = create_manifest(
        config_path=test_config,
        artifact_dir=test_artifact_dir,
        seeds=[42, 1337],
        sign=False
    )

    # Save
    manifest.save(f"{test_artifact_dir}/MANIFEST.json")
    print("✅ Manifest created")

    # Test 2: Verify manifest
    print("\nTest 2: Verify manifest")
    passed = verify_manifest(f"{test_artifact_dir}/MANIFEST.json", verbose=True)
    assert passed
    print("✅ Verification passed")

    print("\n" + "=" * 70)
    print("All tests passed! Run manifest ready for Phase 4.")
    print("=" * 70)
