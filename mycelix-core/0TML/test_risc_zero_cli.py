#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Quick test: RISC Zero CLI verification"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tests/integration"))

from zkbackend_abstraction import get_backend

def main():
    print("Testing RISC Zero backend with correct CLI...")

    # Simple test case: 3 rounds, threshold 1.0 (Q16.16)
    public_inputs = {
        "n": 3,
        "k": 2,
        "m": 1,
        "w": 0,
        "threshold": 65536,  # 1.0 in Q16.16
        "scale": 65536
    }

    # Witness: 2 violations should trigger quarantine
    witness = [50000, 40000, 45000]  # All below threshold

    output_dir = Path("test_risc_output")
    output_dir.mkdir(exist_ok=True)

    try:
        backend = get_backend("risc_zero")
        print(f"✅ Backend loaded: {backend.name()} v{backend.version()}")
        print(f"   Prover path: {backend.prover_path}")

        print("\nGenerating proof...")
        result = backend.prove(public_inputs, witness, output_dir)

        print(f"\n✅ SUCCESS!")
        print(f"   Proving time: {result.proving_time_ms:.0f}ms ({result.proving_time_ms/1000:.1f}s)")
        print(f"   Proof size: {result.proof_size_bytes:,} bytes")
        print(f"   Quarantine decision: {result.quarantine_decision}")

        return 0

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
