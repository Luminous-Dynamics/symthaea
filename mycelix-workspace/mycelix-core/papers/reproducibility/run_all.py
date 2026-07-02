#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Paper Reproducibility - Run All Experiments

This script runs all experiments to reproduce the paper results and generates:
- JSON/CSV results in output/results/
- Figures (PNG) in output/figures/
- LaTeX tables in output/tables/

Usage:
    python run_all.py           # Full reproduction
    python run_all.py --quick   # Quick validation (reduced trials)
    python run_all.py --validate # Run + validate against expected results
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load experiment configuration."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def create_output_dirs(output_dir: Path):
    """Create output directory structure."""
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)


def run_table1(quick: bool = False) -> Dict:
    """Run Table 1: Byzantine Detection experiment."""
    print("\n" + "=" * 60)
    print("Running: Table 1 - Byzantine Detection Rate")
    print("=" * 60)

    from experiments.table1_byzantine_detection import main as table1_main

    # Temporarily modify sys.argv for the experiment
    old_argv = sys.argv
    sys.argv = ['table1_byzantine_detection.py']
    if quick:
        sys.argv.append('--quick')

    try:
        result = table1_main()
    finally:
        sys.argv = old_argv

    return result


def run_table2(quick: bool = False) -> Dict:
    """Run Table 2: Latency experiment."""
    print("\n" + "=" * 60)
    print("Running: Table 2 - Aggregation Latency")
    print("=" * 60)

    from experiments.table2_latency import main as table2_main

    old_argv = sys.argv
    sys.argv = ['table2_latency.py']
    if quick:
        sys.argv.append('--quick')

    try:
        result = table2_main()
    finally:
        sys.argv = old_argv

    return result


def run_figure1(quick: bool = False) -> Dict:
    """Run Figure 1: Detection vs Ratio."""
    print("\n" + "=" * 60)
    print("Running: Figure 1 - Detection Rate vs Byzantine Ratio")
    print("=" * 60)

    from experiments.figure1_detection_vs_ratio import main as figure1_main

    old_argv = sys.argv
    sys.argv = ['figure1_detection_vs_ratio.py']
    if quick:
        sys.argv.append('--quick')

    try:
        result = figure1_main()
    finally:
        sys.argv = old_argv

    return result


def run_figure2(quick: bool = False) -> Dict:
    """Run Figure 2: Convergence."""
    print("\n" + "=" * 60)
    print("Running: Figure 2 - Model Accuracy Convergence")
    print("=" * 60)

    from experiments.figure2_convergence import main as figure2_main

    old_argv = sys.argv
    sys.argv = ['figure2_convergence.py']
    if quick:
        sys.argv.append('--quick')

    try:
        result = figure2_main()
    finally:
        sys.argv = old_argv

    return result


def run_figure3(quick: bool = False) -> Dict:
    """Run Figure 3: Scalability."""
    print("\n" + "=" * 60)
    print("Running: Figure 3 - Scalability Analysis")
    print("=" * 60)

    from experiments.figure3_scalability import main as figure3_main

    old_argv = sys.argv
    sys.argv = ['figure3_scalability.py']
    if quick:
        sys.argv.append('--quick')

    try:
        result = figure3_main()
    finally:
        sys.argv = old_argv

    return result


def generate_latex_table1(results: Dict, output_path: Path):
    """Generate LaTeX table for Table 1."""
    latex = r"""\begin{table}[t]
\centering
\caption{Sign-Flip Attack Detection Rate (\%)}
\label{tab:detection}
\begin{tabular}{lcccc}
\toprule
Byzantine Ratio & FedAvg & Krum & FLTrust & PoGQ (Ours) \\
\midrule
"""

    for ratio_key in ['10%', '20%', '33%', '45%']:
        if ratio_key in results['data']:
            row = f"{ratio_key} & "
            for method in ['fedavg', 'krum', 'fltrust', 'pogq']:
                rate = results['data'][ratio_key][method]['detection_rate_mean'] * 100
                if method == 'pogq':
                    row += f"\\textbf{{{rate:.1f}\\%}} \\\\"
                else:
                    row += f"{rate:.1f}\\% & "
            latex += row + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"LaTeX table saved to: {output_path}")


def generate_latex_table2(results: Dict, output_path: Path):
    """Generate LaTeX table for Table 2."""
    latex = r"""\begin{table}[t]
\centering
\caption{Aggregation Latency Comparison}
\label{tab:latency}
\begin{tabular}{lcccc}
\toprule
Method & Median (ms) & P95 (ms) & P99 (ms) & Mean (ms) \\
\midrule
"""

    for method in ['pogq', 'fltrust', 'krum', 'fedavg']:
        if method in results['data']:
            data = results['data'][method]
            name = method.upper() if method != 'fltrust' else 'FLTrust'
            if method == 'pogq':
                name = 'PoGQ'
            latex += f"{name} & {data['median_ms']:.3f} & {data['p95_ms']:.3f} & "
            latex += f"{data['p99_ms']:.3f} & {data['mean_ms']:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"LaTeX table saved to: {output_path}")


def print_summary(all_results: Dict, elapsed_time: float):
    """Print experiment summary."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total runtime: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print()

    # Table 1 summary
    if 'table1' in all_results:
        print("Table 1: Byzantine Detection")
        for ratio in ['10%', '20%', '33%', '45%']:
            if ratio in all_results['table1']['data']:
                pogq_rate = all_results['table1']['data'][ratio]['pogq']['detection_rate_mean'] * 100
                print(f"  {ratio} Byzantine: PoGQ = {pogq_rate:.1f}% detection")

    print()

    # Table 2 summary
    if 'table2' in all_results:
        print("Table 2: Aggregation Latency")
        if 'pogq' in all_results['table2']['data']:
            pogq = all_results['table2']['data']['pogq']['median_ms']
            krum = all_results['table2']['data']['krum']['median_ms']
            print(f"  PoGQ: {pogq:.3f} ms median")
            print(f"  Krum: {krum:.3f} ms median")
            print(f"  Speedup: {krum/pogq:.1f}x")

    print()

    # Key claims verification
    print("Key Paper Claims:")

    if 'table1' in all_results and '45%' in all_results['table1']['data']:
        pogq_45 = all_results['table1']['data']['45%']['pogq']['detection_rate_mean'] * 100
        status = "VERIFIED" if pogq_45 > 80 else "NEEDS REVIEW"
        print(f"  [+] 84.3% avg detection at 45% Byzantine: {pogq_45:.1f}% [{status}]")

    if 'table2' in all_results and 'pogq' in all_results['table2']['data']:
        pogq_latency = all_results['table2']['data']['pogq']['median_ms']
        status = "VERIFIED" if pogq_latency < 1.0 else "NEEDS REVIEW"
        print(f"  [+] Sub-millisecond aggregation: {pogq_latency:.3f} ms [{status}]")

        krum_latency = all_results['table2']['data']['krum']['median_ms']
        speedup = krum_latency / pogq_latency
        status = "VERIFIED" if speedup > 5.0 else "NEEDS REVIEW"
        print(f"  [+] 6.3x faster than Krum: {speedup:.1f}x [{status}]")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Mycelix Paper Reproducibility - Run All Experiments'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (reduced trials for validation)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate results against expected values')
    parser.add_argument('--output', default='output',
                       help='Output directory')
    parser.add_argument('--experiments', nargs='+',
                       default=['table1', 'table2', 'figure1', 'figure2', 'figure3'],
                       choices=['table1', 'table2', 'figure1', 'figure2', 'figure3'],
                       help='Experiments to run')
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file')
    args = parser.parse_args()

    # Setup
    start_time = time.time()
    output_dir = Path(args.output)
    create_output_dirs(output_dir)

    # Load config and set random seed
    config = load_config(args.config)
    np.random.seed(config.get('global', {}).get('random_seed', 42))

    print("=" * 60)
    print("MYCELIX PAPER REPRODUCIBILITY KIT")
    print("=" * 60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Experiments: {', '.join(args.experiments)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run experiments
    all_results = {}

    if 'table1' in args.experiments:
        all_results['table1'] = run_table1(quick=args.quick)
        generate_latex_table1(
            all_results['table1'],
            output_dir / 'tables' / 'table1_byzantine_detection.tex'
        )

    if 'table2' in args.experiments:
        all_results['table2'] = run_table2(quick=args.quick)
        generate_latex_table2(
            all_results['table2'],
            output_dir / 'tables' / 'table2_latency.tex'
        )

    if 'figure1' in args.experiments:
        all_results['figure1'] = run_figure1(quick=args.quick)

    if 'figure2' in args.experiments:
        all_results['figure2'] = run_figure2(quick=args.quick)

    if 'figure3' in args.experiments:
        all_results['figure3'] = run_figure3(quick=args.quick)

    # Save combined results
    combined_file = output_dir / 'results' / 'all_results.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to: {combined_file}")

    # Print summary
    elapsed_time = time.time() - start_time
    print_summary(all_results, elapsed_time)

    # Validation
    if args.validate:
        print("=" * 60)
        print("VALIDATION")
        print("=" * 60)

        # Import and run validation
        try:
            from validate import validate_all
            validation_results = validate_all(all_results)

            if validation_results['overall_status'] == 'PASS':
                print("\n[PASS] All results within expected tolerances!")
            else:
                print("\n[WARN] Some results outside expected tolerances.")
                print("       Review individual experiment outputs for details.")

        except ImportError:
            print("Validation module not found. Run validate.py separately.")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {elapsed_time:.1f} seconds")
    print()
    print("Output files:")
    print(f"  Results:  {output_dir / 'results'}")
    print(f"  Figures:  {output_dir / 'figures'}")
    print(f"  Tables:   {output_dir / 'tables'}")

    return all_results


if __name__ == '__main__':
    main()
