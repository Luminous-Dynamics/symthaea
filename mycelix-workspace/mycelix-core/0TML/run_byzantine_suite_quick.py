#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Quick Byzantine Attack Suite

Runs reduced test (10 rounds instead of 100) for rapid validation.
Total time: ~3-5 hours instead of 58 hours.

Usage:
    python run_byzantine_suite_quick.py
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
    """Run quick Byzantine validation suite"""
    print("=" * 70)
    print("🚀 QUICK BYZANTINE VALIDATION SUITE")
    print("=" * 70)
    print("Configuration:")
    print("  - 10 rounds (instead of 100)")
    print("  - All 7 attack types")
    print("  - All 5 defense mechanisms")
    print("  - Total: 35 experiments")
    print("  - Estimated time: ~3-5 hours")
    print()

    # Use the quick config
    base_config = "experiments/configs/mnist_byzantine_attacks_quick.yaml"

    if not Path(base_config).exists():
        print(f"❌ Config not found: {base_config}")
        return 1

    # Create runner with quick config
    runner = ByzantineSuiteRunner(base_config)

    # Override results directory for quick tests
    runner.results_dir = Path("/tmp/byzantine_results_quick")
    runner.results_dir.mkdir(exist_ok=True)

    print(f"📁 Results will be saved to: {runner.results_dir}")
    print()

    # Run all experiments
    success = runner.run_all()

    return 0 if success else 1


if __name__ == "__main__":
    print("=" * 70)
    print("QUICK BYZANTINE SUITE STARTING")
    print("=" * 70)
    sys.exit(main())
