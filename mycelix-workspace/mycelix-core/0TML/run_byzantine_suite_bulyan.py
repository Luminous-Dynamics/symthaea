#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Bulyan-Specific Byzantine Attack Suite

Tests Bulyan defense with theory-compliant Byzantine ratio (f=2 < n/3).
This validates Bulyan's theoretical guarantees.

Total experiments: 7 (all attacks × Bulyan only)
Total time: ~30-45 minutes on GPU

Usage:
    python run_byzantine_suite_bulyan.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the full suite runner
from run_byzantine_suite import ByzantineSuiteRunner

def main():
    """Run Bulyan-specific Byzantine validation suite"""
    print("=" * 70)
    print("🛡️  BULYAN THEORY-COMPLIANT VALIDATION SUITE")
    print("=" * 70)
    print("Configuration:")
    print("  - 2 Byzantine clients (20%, theory-compliant f < n/3)")
    print("  - All 7 attack types")
    print("  - Bulyan defense only")
    print("  - 10 rounds per experiment")
    print("  - Total: 7 experiments")
    print("  - Estimated time: ~30-45 minutes")
    print()
    print("Research Goal:")
    print("  Validate Bulyan's theoretical guarantees when f=2 < n/3")
    print("  Compare against high-stress scenario (30% Byzantine)")
    print()

    # Use the Bulyan-specific config
    base_config = "experiments/configs/mnist_byzantine_attacks_bulyan.yaml"

    if not Path(base_config).exists():
        print(f"❌ Config not found: {base_config}")
        return 1

    # Create runner with Bulyan config
    runner = ByzantineSuiteRunner(base_config)

    # Override results directory for Bulyan tests
    runner.results_dir = Path("/tmp/byzantine_results_bulyan")
    runner.results_dir.mkdir(exist_ok=True)

    print(f"📁 Results will be saved to: {runner.results_dir}")
    print()

    # Run all experiments
    success = runner.run_all()

    if success:
        print()
        print("=" * 70)
        print("✅ BULYAN SUITE COMPLETE!")
        print("=" * 70)
        print(f"Results available at: {runner.results_dir}")
        print()
        print("Next steps:")
        print("  1. Compare Bulyan performance: 20% vs 30% Byzantine")
        print("  2. Validate theoretical guarantees held")
        print("  3. Design PoGQ + Reputation baseline")
        print()

    return 0 if success else 1

if __name__ == "__main__":
    print("=" * 70)
    print("BULYAN BYZANTINE SUITE STARTING")
    print("=" * 70)
    sys.exit(main())
