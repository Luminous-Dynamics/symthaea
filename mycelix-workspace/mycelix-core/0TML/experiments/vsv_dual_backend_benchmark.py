#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Dual-Backend VSV-STARK Benchmarking Suite

Compares RISC Zero zkVM vs Winterfell AIR performance for PoGQ proving.

Target Performance (Winterfell):
  - Prove time: 5-15 seconds (3-10× faster than RISC Zero)
  - Verify time: <50ms
  - Proof size: ~200KB (similar to RISC Zero)

Usage:
  python experiments/vsv_dual_backend_benchmark.py --rounds 5 --output results/dual_backend_comparison.json
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_risc_zero_prover() -> Path:
    """Find RISC Zero prover binary"""
    candidates = [
        Path("/tmp/vsv-prover/release/host"),
        Path("vsv-stark/target/release/host"),
        Path("target/release/host"),
    ]

    for path in candidates:
        if path.exists():
            logger.info(f"✅ RISC Zero prover found: {path}")
            logger.info(f"   Size: {path.stat().st_size / 1024 / 1024:.1f} MB")
            return path

    raise FileNotFoundError("RISC Zero prover not found. Run: cd vsv-stark && cargo build --release")


def find_winterfell_prover() -> Path:
    """Find or build Winterfell prover binary"""
    binary = Path("vsv-stark/target/release/winterfell-prover")

    if not binary.exists():
        logger.info("🔨 Building Winterfell prover...")
        subprocess.run(
            ["cargo", "build", "--release", "-p", "winterfell-pogq"],
            cwd="vsv-stark",
            check=True,
            capture_output=True,
        )

    if binary.exists():
        logger.info(f"✅ Winterfell prover found: {binary}")
        logger.info(f"   Size: {binary.stat().st_size / 1024 / 1024:.1f} MB")
        return binary

    raise FileNotFoundError("Failed to build Winterfell prover")


def create_pogq_inputs(round_num: int) -> Tuple[Dict, Dict]:
    """Create realistic PoGQ public/witness inputs"""
    # Realistic PoGQ parameters from experiments
    beta = 0.85  # EMA smoothing
    w = 10  # Warm-up rounds
    k = 3  # Violations to quarantine
    m = 5  # Clears to release
    threshold = 0.10  # Conformal threshold

    # Simulate state progression
    ema_init = 0.85 + (round_num * 0.01) % 0.15  # Drift slightly
    viol_init = round_num % 2  # Occasional violation
    clear_init = 2 if viol_init == 0 else 0
    quar_init = 1 if round_num % 7 == 0 else 0  # Occasional quarantine
    round_init = round_num

    # Generate witness (hybrid scores)
    # Power-of-2 trace length (Winterfell requirement)
    trace_length = 8
    import random

    random.seed(42 + round_num)  # Deterministic
    scores = [random.uniform(0.80, 0.95) for _ in range(trace_length)]

    # Compute expected output
    # Simplified: assume no quarantine for demonstration
    quar_out = 0

    public = {
        "beta": beta,
        "w": w,
        "k": k,
        "m": m,
        "threshold": threshold,
        "ema_init": ema_init,
        "viol_init": viol_init,
        "clear_init": clear_init,
        "quar_init": quar_init,
        "round_init": round_init,
        "quar_out": quar_out,
        "trace_length": trace_length,
    }

    witness = {"scores": scores}

    return public, witness


def benchmark_risc_zero(prover_path: Path, public: Dict, witness: Dict) -> Dict:
    """Benchmark RISC Zero zkVM prover"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write inputs
        public_file = Path(tmpdir) / "public.json"
        witness_file = Path(tmpdir) / "witness.json"
        proof_file = Path(tmpdir) / "proof.bin"

        public_file.write_text(json.dumps(public))
        witness_file.write_text(json.dumps(witness))

        # Run prover
        start = time.time()
        result = subprocess.run(
            [
                str(prover_path),
                "--public",
                str(public_file),
                "--witness",
                str(witness_file),
                "--output",
                str(proof_file),
            ],
            capture_output=True,
            text=True,
        )
        elapsed_ms = (time.time() - start) * 1000

        if result.returncode != 0:
            logger.error(f"RISC Zero prover failed: {result.stderr}")
            return {"success": False, "error": result.stderr}

        # Read proof
        proof_bytes = proof_file.read_bytes()

        return {
            "success": True,
            "prove_ms": elapsed_ms,
            "proof_size_bytes": len(proof_bytes),
            "stdout": result.stdout,
        }


def benchmark_winterfell(prover_path: Path, public: Dict, witness: Dict) -> Dict:
    """Benchmark Winterfell AIR prover"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write inputs
        public_file = Path(tmpdir) / "public.json"
        witness_file = Path(tmpdir) / "witness.json"
        proof_file = Path(tmpdir) / "proof.bin"

        public_file.write_text(json.dumps(public))
        witness_file.write_text(json.dumps(witness))

        # Run prover
        start = time.time()
        result = subprocess.run(
            [
                str(prover_path),
                "prove",
                "--public",
                str(public_file),
                "--witness",
                str(witness_file),
                "--output",
                str(proof_file),
            ],
            capture_output=True,
            text=True,
        )
        elapsed_ms = (time.time() - start) * 1000

        if result.returncode != 0:
            logger.error(f"Winterfell prover failed: {result.stderr}")
            return {"success": False, "error": result.stderr}

        # Read proof
        proof_bytes = proof_file.read_bytes()

        # Parse timing from output
        # Expected format: "Total: XXXms"
        prove_ms = elapsed_ms  # Default to total time
        for line in result.stdout.split("\n"):
            if "Proving:" in line:
                try:
                    prove_ms = float(line.split(":")[1].strip().replace("ms", ""))
                except:
                    pass

        return {
            "success": True,
            "prove_ms": prove_ms,
            "total_ms": elapsed_ms,
            "proof_size_bytes": len(proof_bytes),
            "stdout": result.stdout,
        }


def main():
    parser = argparse.ArgumentParser(description="Dual-backend VSV-STARK benchmark")
    parser.add_argument(
        "--rounds", type=int, default=3, help="Number of proving rounds"
    )
    parser.add_argument(
        "--output", default="results/dual_backend_comparison.json", help="Output JSON"
    )
    parser.add_argument(
        "--skip-risc-zero", action="store_true", help="Skip RISC Zero (test Winterfell only)"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("🔬 VSV-STARK DUAL-BACKEND BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Rounds: {args.rounds}")
    logger.info(f"Output: {args.output}")
    logger.info("")

    # Find provers
    winterfell_prover = find_winterfell_prover()
    risc_zero_prover = None
    if not args.skip_risc_zero:
        try:
            risc_zero_prover = find_risc_zero_prover()
        except FileNotFoundError as e:
            logger.warning(f"⚠️  RISC Zero prover not found: {e}")
            logger.warning("   Continuing with Winterfell only")

    results = {
        "metadata": {
            "rounds": args.rounds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "winterfell_prover": str(winterfell_prover),
            "risc_zero_prover": str(risc_zero_prover) if risc_zero_prover else None,
        },
        "winterfell": [],
        "risc_zero": [],
    }

    # Run benchmarks
    for i in range(args.rounds):
        logger.info(f"\n📊 Round {i+1}/{args.rounds}")

        # Generate inputs
        public, witness = create_pogq_inputs(i)
        logger.info(f"   Inputs: β={public['beta']}, trace_len={public['trace_length']}")

        # Winterfell
        logger.info("   🔬 Winterfell AIR...")
        wf_result = benchmark_winterfell(winterfell_prover, public, witness)
        results["winterfell"].append(wf_result)

        if wf_result["success"]:
            logger.info(
                f"      ✅ {wf_result['prove_ms']:.1f}ms, {wf_result['proof_size_bytes']} bytes"
            )
        else:
            logger.error(f"      ❌ Failed: {wf_result.get('error', 'Unknown')}")

        # RISC Zero
        if risc_zero_prover:
            logger.info("   🔬 RISC Zero zkVM...")
            rz_result = benchmark_risc_zero(risc_zero_prover, public, witness)
            results["risc_zero"].append(rz_result)

            if rz_result["success"]:
                logger.info(
                    f"      ✅ {rz_result['prove_ms']:.1f}ms, {rz_result['proof_size_bytes']} bytes"
                )
            else:
                logger.error(f"      ❌ Failed: {rz_result.get('error', 'Unknown')}")

    # Compute statistics
    logger.info("\n" + "=" * 70)
    logger.info("📊 SUMMARY STATISTICS")
    logger.info("=" * 70)

    wf_times = [r["prove_ms"] for r in results["winterfell"] if r["success"]]
    if wf_times:
        logger.info("\nWinterfell AIR:")
        logger.info(f"  Avg prove time: {sum(wf_times) / len(wf_times):.1f}ms")
        logger.info(f"  Min: {min(wf_times):.1f}ms")
        logger.info(f"  Max: {max(wf_times):.1f}ms")

    rz_times = [r["prove_ms"] for r in results["risc_zero"] if r["success"]]
    if rz_times:
        logger.info("\nRISC Zero zkVM:")
        logger.info(f"  Avg prove time: {sum(rz_times) / len(rz_times):.1f}ms")
        logger.info(f"  Min: {min(rz_times):.1f}ms")
        logger.info(f"  Max: {max(rz_times):.1f}ms")

    if wf_times and rz_times:
        speedup = (sum(rz_times) / len(rz_times)) / (sum(wf_times) / len(wf_times))
        logger.info(f"\n⚡ Speedup: {speedup:.2f}x (Winterfell vs RISC Zero)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\n💾 Results saved: {output_path}")


if __name__ == "__main__":
    main()
