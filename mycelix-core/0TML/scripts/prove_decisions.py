#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
VSV-STARK: Prove PoGQ Decision Logic
=====================================

Generates STARK proofs for PoGQ-v4.1 per-round decision logic.

Usage:
    python scripts/prove_decisions.py results/artifacts_run_001/

This will:
1. Load artifacts (detection_metrics.json, model_metrics.json)
2. Extract decision parameters (EMA, hysteresis, quarantine)
3. Generate STARK proof for one representative round
4. Save proof to decision_proof.bin

Author: Luminous Dynamics
Date: November 8, 2025
"""

import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional

# Check if vsv_stark module is available
try:
    import vsv_stark
    STARK_AVAILABLE = True
except ImportError:
    STARK_AVAILABLE = False
    print("⚠️  vsv_stark module not installed")
    print("   Build with: cd verif/python && maturin develop")
    print("")

def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_artifacts(artifact_dir: Path) -> Dict:
    """Load all artifacts from directory."""
    artifacts = {}

    # Required files
    required_files = [
        'detection_metrics.json',
        'per_bucket_fpr.json',
        'ema_vs_raw.json',
        'coord_median_diagnostics.json',
        'bootstrap_ci.json',
        'model_metrics.json'
    ]

    for filename in required_files:
        file_path = artifact_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing artifact: {filename}")

        with open(file_path, 'r') as f:
            artifacts[filename] = json.load(f)

    return artifacts

def extract_decision_parameters(artifacts: Dict) -> Tuple[Dict, Dict]:
    """
    Extract public inputs and witness from artifacts.

    Returns:
        (public_inputs, witness)
    """
    # For v0, we'll use synthetic values since we don't store
    # per-round decision states in current artifacts.
    # In production, these would come from detector logs.

    # Hash commitments (placeholder - would hash actual calibration/model/grad)
    h_calib = "a" * 64  # Placeholder
    h_model = "b" * 64  # Placeholder
    h_grad = "c" * 64   # Placeholder

    # PoGQ parameters (from Phase 2 config)
    beta_fp = vsv_stark.to_fixed(0.85)  # EMA smoothing
    w = 3          # Warm-up rounds
    k = 2          # Violations to quarantine
    m = 3          # Clears to release
    egregious_cap_fp = vsv_stark.to_fixed(0.9999)

    # Conformal threshold (from per_bucket_fpr.json)
    per_bucket = artifacts['per_bucket_fpr.json']
    # Use first bucket's FPR as proxy (in production, use actual threshold)
    if per_bucket['buckets']:
        threshold = per_bucket['buckets'][0]['fpr']
    else:
        threshold = 0.10  # Default α

    threshold_fp = vsv_stark.to_fixed(threshold)

    # Previous state (synthetic for v0)
    # In production, this comes from detector's state tracking
    ema_prev_fp = vsv_stark.to_fixed(0.75)
    consec_viol_prev = 1
    consec_clear_prev = 0
    quarantined_prev = 0
    current_round = 5

    # Expected output (synthetic)
    # In production, this is the actual quarantine decision
    quarantine_out = 0

    public_inputs = {
        "h_calib": h_calib,
        "h_model": h_model,
        "h_grad": h_grad,
        "beta_fp": beta_fp,
        "w": w,
        "k": k,
        "m": m,
        "egregious_cap_fp": egregious_cap_fp,
        "threshold_fp": threshold_fp,
        "ema_prev_fp": ema_prev_fp,
        "consec_viol_prev": consec_viol_prev,
        "consec_clear_prev": consec_clear_prev,
        "quarantined_prev": quarantined_prev,
        "current_round": current_round,
        "quarantine_out": quarantine_out,
    }

    # Private witness (synthetic)
    # In production, this is the actual hybrid score
    x_t_fp = vsv_stark.to_fixed(0.80)
    in_warmup = 0
    violation_t = 1 if x_t_fp < threshold_fp else 0
    release_t = 0

    witness = {
        "x_t_fp": x_t_fp,
        "in_warmup": in_warmup,
        "violation_t": violation_t,
        "release_t": release_t,
    }

    return public_inputs, witness

def generate_proof(artifact_dir: Path) -> Optional[Path]:
    """
    Generate STARK proof for decision logic.

    Returns:
        Path to proof file if successful, None otherwise
    """
    if not STARK_AVAILABLE:
        print("❌ Cannot generate proof: vsv_stark module not installed")
        return None

    print("=" * 70)
    print("VSV-STARK: Generating Decision Proof")
    print("=" * 70)
    print("")

    # Load artifacts
    print(f"📂 Loading artifacts from {artifact_dir.name}...")
    try:
        artifacts = load_artifacts(artifact_dir)
        print(f"   ✅ Loaded {len(artifacts)} artifact files")
    except Exception as e:
        print(f"   ❌ Failed to load artifacts: {e}")
        return None

    print("")

    # Extract parameters
    print("🔍 Extracting decision parameters...")
    try:
        public_inputs, witness = extract_decision_parameters(artifacts)
        print(f"   ✅ Public inputs: {len(public_inputs)} fields")
        print(f"   ✅ Witness: {len(witness)} fields")

        # Display key values
        print("")
        print("   Key parameters:")
        print(f"     - EMA beta: {vsv_stark.from_fixed(public_inputs['beta_fp']):.4f}")
        print(f"     - Threshold: {vsv_stark.from_fixed(public_inputs['threshold_fp']):.4f}")
        print(f"     - Hybrid score: {vsv_stark.from_fixed(witness['x_t_fp']):.4f}")
        print(f"     - Round: {public_inputs['current_round']}")
        print(f"     - Expected output: quarantine={public_inputs['quarantine_out']}")
    except Exception as e:
        print(f"   ❌ Failed to extract parameters: {e}")
        return None

    print("")

    # Generate proof
    print("⚙️  Generating STARK proof...")
    try:
        proof_bytes, proof_hex = vsv_stark.generate_round_proof(
            public_inputs,
            witness
        )
        print(f"   ✅ Proof generated ({len(proof_bytes)} bytes)")
        print(f"   ✅ Proof hash: {proof_hex[:16]}...")
    except Exception as e:
        print(f"   ❌ Proof generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("")

    # Verify proof
    print("🔐 Verifying proof...")
    try:
        is_valid = vsv_stark.verify_round_proof(
            proof_bytes,
            json.dumps(public_inputs)
        )
        if is_valid:
            print("   ✅ Proof verified successfully")
        else:
            print("   ❌ Proof verification failed")
            return None
    except Exception as e:
        print(f"   ❌ Verification error: {e}")
        return None

    print("")

    # Save proof
    proof_path = artifact_dir / "decision_proof.bin"
    print(f"💾 Saving proof to {proof_path.name}...")
    try:
        proof_path.write_bytes(proof_bytes)
        print(f"   ✅ Proof saved ({len(proof_bytes)} bytes)")

        # Also save public inputs
        public_path = artifact_dir / "decision_public.json"
        with open(public_path, 'w') as f:
            json.dump(public_inputs, f, indent=2)
        print(f"   ✅ Public inputs saved to {public_path.name}")
    except Exception as e:
        print(f"   ❌ Failed to save proof: {e}")
        return None

    print("")
    print("=" * 70)
    print("✅ Proof generation complete")
    print("=" * 70)

    return proof_path

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate STARK proofs for PoGQ decision logic'
    )
    parser.add_argument(
        'artifact_dir',
        type=Path,
        help='Directory containing artifacts (e.g., results/artifacts_run_001/)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing proof, do not generate new one'
    )

    args = parser.parse_args()

    # Verify directory exists
    if not args.artifact_dir.exists():
        print(f"❌ Directory not found: {args.artifact_dir}")
        sys.exit(1)

    if not args.artifact_dir.is_dir():
        print(f"❌ Not a directory: {args.artifact_dir}")
        sys.exit(1)

    # Verify-only mode
    if args.verify_only:
        proof_path = args.artifact_dir / "decision_proof.bin"
        public_path = args.artifact_dir / "decision_public.json"

        if not proof_path.exists():
            print(f"❌ Proof not found: {proof_path}")
            sys.exit(1)

        if not public_path.exists():
            print(f"❌ Public inputs not found: {public_path}")
            sys.exit(1)

        print("🔐 Verifying existing proof...")
        proof_bytes = proof_path.read_bytes()

        with open(public_path, 'r') as f:
            public_inputs = json.load(f)

        try:
            is_valid = vsv_stark.verify_round_proof(
                proof_bytes,
                json.dumps(public_inputs)
            )
            if is_valid:
                print("✅ Proof verified successfully")
                sys.exit(0)
            else:
                print("❌ Proof verification failed")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Verification error: {e}")
            sys.exit(1)

    # Generate proof
    proof_path = generate_proof(args.artifact_dir)

    if proof_path is None:
        sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()
