# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Full 0TML Hybrid Detector at 30% BFT (Re-validation)

This test re-validates our previous 30% BFT success with the current
detector configuration to establish a baseline before tuning.

Previous Result (from earlier tests):
- Detection Rate: 83.3% (5/6)
- False Positive Rate: 0.0% (0/14)

Expected with current detector:
- If still passes: Confirms 30-35% is the tipping point
- If now fails (0% detection): Confirms threshold miscalibration

Author: Zero-TrustML Research Team
Date: November 5, 2025
Status: Baseline re-validation before threshold tuning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_full_hybrid_35bft import FullHybridDetectorTest

def test_full_hybrid_30_bft():
    """
    Test Full 0TML Hybrid Detector at 30% BFT.

    This establishes whether our detector works at 30% with current configuration.
    """
    test = FullHybridDetectorTest(
        num_clients=20,
        num_byzantine=6,  # 30% BFT
        test_name="Full 0TML Hybrid Detector at 30% BFT (Re-validation)",
        seed=42
    )

    # Run for 3 rounds
    results = test.run_test(num_rounds=3)

    print("\n" + "="*80)
    print("📊 30% BFT RE-VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  Detection Rate: {results['detection_rate']*100:.1f}%")
    print(f"  False Positive Rate: {results['fpr']*100:.1f}%")
    print(f"  Test Passed: {results['test_passed']}")

    if results['detection_rate'] == 0.0:
        print("\n⚠️  CRITICAL: 0% detection at 30% BFT confirms threshold miscalibration")
        print("   Previous tests showed 83.3% detection - threshold is too conservative")
    elif results['detection_rate'] >= 0.8:
        print("\n✅ SUCCESS: Detector still works at 30% BFT")
        print("   This confirms 30-35% BFT range is the tipping point")
    else:
        print(f"\n⚠️  PARTIAL: {results['detection_rate']*100:.1f}% detection at 30% BFT")
        print("   Detector degraded but not completely failed")

    print("="*80 + "\n")

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔬 30% BFT RE-VALIDATION - ESTABLISHING BASELINE")
    print("="*80 + "\n")

    print("Purpose: Determine if 30% BFT success is still reproducible")
    print("Configuration: Same as 35% BFT test but with 6 Byzantine (30%)\n")

    results = test_full_hybrid_30_bft()

    print("Next Step: Threshold parameter sweep to find optimal value\n")
