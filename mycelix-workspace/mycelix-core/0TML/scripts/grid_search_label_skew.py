#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Grid Search for Optimal Label Skew Parameters

Automatically explores parameter combinations to minimize false positive rate
while maintaining ≥95% detection rate.

Usage:
    python scripts/grid_search_label_skew.py [--quick] [--output results.json]
"""

import subprocess
import json
import itertools
import sys
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse


class GridSearchOptimizer:
    """Automated parameter optimization for label skew detection."""

    def __init__(self, output_path: str = "results/grid_search_results.json"):
        self.output_path = Path(output_path)
        self.results: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None

    def define_search_space(self, quick: bool = False) -> Dict[str, List]:
        """Define parameter ranges to explore."""
        if quick:
            # Quick search: ~16 combinations (5-10 minutes)
            return {
                'BEHAVIOR_RECOVERY_THRESHOLD': [3, 4],
                'BEHAVIOR_RECOVERY_BONUS': [0.10, 0.12],
                'LABEL_SKEW_COS_MIN': [-0.3, -0.4],
                'LABEL_SKEW_COS_MAX': [0.95, 0.97],
            }
        else:
            # Full search: ~192 combinations (4-6 hours)
            return {
                'BEHAVIOR_RECOVERY_THRESHOLD': [2, 3, 4, 5],
                'BEHAVIOR_RECOVERY_BONUS': [0.06, 0.08, 0.10, 0.12, 0.15],
                'LABEL_SKEW_COS_MIN': [-0.25, -0.3, -0.35, -0.4],
                'LABEL_SKEW_COS_MAX': [0.93, 0.95, 0.97],
            }

    def run_test(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run single test with given parameters."""
        print(f"\n{'='*70}")
        print(f"Testing: {params}")
        print(f"{'='*70}")

        # Set up environment
        env = os.environ.copy()
        env.update({
            'RUN_30_BFT': '1',
            'BFT_DISTRIBUTION': 'label_skew',
            'LABEL_SKEW_TRACE_PATH': '',  # Disable trace to speed up
            **{k: str(v) for k, v in params.items()}
        })

        try:
            # Run test (10 minute timeout)
            cmd = ['poetry', 'run', 'python', 'tests/test_30_bft_validation.py']
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
                cwd='/srv/luminous-dynamics/Mycelix-Core/0TML'
            )

            # Parse output for metrics
            output = result.stdout + result.stderr
            metrics = self.parse_metrics(output)

            if metrics:
                metrics['params'] = params
                metrics['success'] = result.returncode == 0
                return metrics
            else:
                print(f"❌ Failed to parse metrics from output")
                return None

        except subprocess.TimeoutExpired:
            print(f"⏱️  Test timed out (>10 minutes)")
            return None
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return None

    def parse_metrics(self, output: str) -> Optional[Dict[str, float]]:
        """Extract metrics from test output."""
        metrics = {}

        # Extract detection rate
        detection_match = re.search(r'Byzantine Detection Rate:\s*([\d.]+)%', output)
        if detection_match:
            metrics['detection_rate'] = float(detection_match.group(1)) / 100

        # Extract false positive rate
        fp_match = re.search(r'False Positive Rate:\s*([\d.]+)%', output)
        if fp_match:
            metrics['fp_rate'] = float(fp_match.group(1)) / 100

        # Extract average honest reputation
        honest_rep_match = re.search(r'Average Honest Reputation:\s*([\d.]+)', output)
        if honest_rep_match:
            metrics['honest_rep'] = float(honest_rep_match.group(1))

        # Return None if we didn't find all metrics
        if len(metrics) >= 2:  # At least detection_rate and fp_rate
            return metrics
        else:
            return None

    def evaluate_result(self, result: Dict[str, Any]) -> float:
        """
        Score a result (lower is better).

        Primary: Minimize FP rate
        Secondary: Maximize detection rate (penalize if <95%)
        Tertiary: Maximize honest reputation
        """
        if not result or not result.get('success'):
            return 999.0  # Worst score for failed tests

        fp_rate = result.get('fp_rate', 1.0)
        detection_rate = result.get('detection_rate', 0.0)
        honest_rep = result.get('honest_rep', 0.0)

        # Heavy penalty if detection rate < 95%
        if detection_rate < 0.95:
            return fp_rate + (0.95 - detection_rate) * 10

        # Otherwise, minimize FP rate with slight bonus for higher honest rep
        return fp_rate - (honest_rep * 0.01)

    def run_grid_search(self, quick: bool = False):
        """Execute grid search over parameter space."""
        search_space = self.define_search_space(quick=quick)

        # Generate all parameter combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        combinations = list(itertools.product(*param_values))

        total = len(combinations)
        print(f"\n🔍 Grid Search Starting")
        print(f"   Parameter space: {total} combinations")
        print(f"   Mode: {'Quick' if quick else 'Full'}")
        print(f"   Estimated time: {total * 0.5:.0f}-{total * 2:.0f} minutes")
        print(f"   Output: {self.output_path}")
        print(f"\n{'='*70}\n")

        # Test each combination
        for i, param_tuple in enumerate(combinations, 1):
            params = dict(zip(param_names, param_tuple))

            print(f"\n[{i}/{total}] Testing combination {i}...")

            result = self.run_test(params)

            if result:
                score = self.evaluate_result(result)
                result['score'] = score
                self.results.append(result)

                # Update best result
                if self.best_result is None or score < self.best_result['score']:
                    self.best_result = result
                    print(f"\n✨ NEW BEST RESULT!")
                    print(f"   FP Rate: {result['fp_rate']*100:.1f}%")
                    print(f"   Detection Rate: {result['detection_rate']*100:.1f}%")
                    print(f"   Honest Rep: {result['honest_rep']:.3f}")
                    print(f"   Params: {result['params']}")

                # Save intermediate results
                self.save_results()
            else:
                print(f"   Skipping failed test")

        print(f"\n{'='*70}")
        print(f"🎉 Grid Search Complete!")
        print(f"{'='*70}\n")

        self.print_summary()

    def save_results(self):
        """Save results to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'best_result': self.best_result,
            'all_results': sorted(self.results, key=lambda x: x['score'])[:20],  # Top 20
        }

        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)

    def print_summary(self):
        """Print summary of grid search results."""
        if not self.best_result:
            print("❌ No successful results found")
            return

        print(f"\n📊 BEST PARAMETERS FOUND:\n")
        print(f"   FP Rate: {self.best_result['fp_rate']*100:.1f}%")
        print(f"   Detection Rate: {self.best_result['detection_rate']*100:.1f}%")
        print(f"   Honest Reputation: {self.best_result['honest_rep']:.3f}")
        print(f"\n   Parameters:")
        for param, value in self.best_result['params'].items():
            print(f"      {param}={value}")

        print(f"\n📈 TOP 5 CONFIGURATIONS:\n")
        top_5 = sorted(self.results, key=lambda x: x['score'])[:5]
        for i, result in enumerate(top_5, 1):
            print(f"   {i}. FP={result['fp_rate']*100:.1f}% | "
                  f"Det={result['detection_rate']*100:.1f}% | "
                  f"Rep={result['honest_rep']:.2f} | "
                  f"Score={result['score']:.3f}")
            print(f"      {result['params']}")

        print(f"\n💾 Full results saved to: {self.output_path}")

        print(f"\n🚀 TO APPLY BEST PARAMETERS:")
        print(f"\n   export BEHAVIOR_RECOVERY_THRESHOLD={self.best_result['params']['BEHAVIOR_RECOVERY_THRESHOLD']}")
        print(f"   export BEHAVIOR_RECOVERY_BONUS={self.best_result['params']['BEHAVIOR_RECOVERY_BONUS']}")
        print(f"   export LABEL_SKEW_COS_MIN={self.best_result['params']['LABEL_SKEW_COS_MIN']}")
        print(f"   export LABEL_SKEW_COS_MAX={self.best_result['params']['LABEL_SKEW_COS_MAX']}")
        print(f"\n   python tests/test_30_bft_validation.py\n")


def main():
    parser = argparse.ArgumentParser(description='Grid search for label skew parameters')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick search (~16 combinations, 5-10 minutes)')
    parser.add_argument('--output', default='results/grid_search_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    optimizer = GridSearchOptimizer(output_path=args.output)
    optimizer.run_grid_search(quick=args.quick)


if __name__ == '__main__':
    main()
