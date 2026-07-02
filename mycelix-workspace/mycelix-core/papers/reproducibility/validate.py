#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Paper Reproducibility - Result Validation

Validates experimental results against expected values from the paper.
Reports any discrepancies with tolerance-aware comparisons.

Usage:
    python validate.py                           # Validate existing results
    python validate.py --results output/results  # Specify results directory
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_expected_results(expected_dir: Path) -> Dict:
    """Load expected results from JSON files."""
    expected = {}

    table1_path = expected_dir / 'table1_expected.json'
    if table1_path.exists():
        with open(table1_path, 'r') as f:
            expected['table1'] = json.load(f)

    table2_path = expected_dir / 'table2_expected.json'
    if table2_path.exists():
        with open(table2_path, 'r') as f:
            expected['table2'] = json.load(f)

    return expected


def load_actual_results(results_dir: Path) -> Dict:
    """Load actual experiment results."""
    actual = {}

    table1_path = results_dir / 'table1_byzantine_detection.json'
    if table1_path.exists():
        with open(table1_path, 'r') as f:
            actual['table1'] = json.load(f)

    table2_path = results_dir / 'table2_latency.json'
    if table2_path.exists():
        with open(table2_path, 'r') as f:
            actual['table2'] = json.load(f)

    # Also try to load from all_results.json
    all_results_path = results_dir / 'all_results.json'
    if all_results_path.exists():
        with open(all_results_path, 'r') as f:
            all_results = json.load(f)
            if 'table1' in all_results and 'table1' not in actual:
                actual['table1'] = all_results['table1']
            if 'table2' in all_results and 'table2' not in actual:
                actual['table2'] = all_results['table2']

    return actual


def validate_table1(actual: Dict, expected: Dict) -> Dict:
    """Validate Table 1 (Byzantine Detection) results."""
    results = {
        'name': 'Table 1: Byzantine Detection',
        'status': 'PASS',
        'checks': [],
        'warnings': []
    }

    if 'data' not in actual:
        results['status'] = 'FAIL'
        results['checks'].append({
            'name': 'Data presence',
            'status': 'FAIL',
            'message': 'No data found in actual results'
        })
        return results

    expected_data = expected.get('data', {})
    tolerance = expected.get('tolerance_percent', 5.0) / 100.0

    for ratio_key in ['10%', '20%', '33%', '45%']:
        if ratio_key not in actual['data']:
            continue

        for method in ['fedavg', 'krum', 'fltrust', 'pogq']:
            if method not in actual['data'][ratio_key]:
                continue
            if ratio_key not in expected_data:
                continue
            if method not in expected_data[ratio_key]:
                continue

            actual_rate = actual['data'][ratio_key][method]['detection_rate_mean']
            expected_rate = expected_data[ratio_key][method]['detection_rate']
            method_tolerance = expected_data[ratio_key][method].get('tolerance', tolerance)

            diff = abs(actual_rate - expected_rate)
            status = 'PASS' if diff <= method_tolerance else 'WARN'

            if status == 'WARN':
                results['status'] = 'WARN'
                results['warnings'].append(
                    f"{method} at {ratio_key}: expected {expected_rate*100:.1f}%, "
                    f"got {actual_rate*100:.1f}% (diff: {diff*100:.1f}%)"
                )

            results['checks'].append({
                'name': f'{method} @ {ratio_key}',
                'status': status,
                'expected': expected_rate,
                'actual': actual_rate,
                'tolerance': method_tolerance,
                'diff': diff
            })

    # Validate key claims
    claims = expected.get('key_claims', {})

    if 'pogq_beats_fltrust_at_45_percent' in claims:
        if '45%' in actual['data']:
            pogq_rate = actual['data']['45%']['pogq']['detection_rate_mean']
            fltrust_rate = actual['data']['45%']['fltrust']['detection_rate_mean']
            margin = pogq_rate - fltrust_rate
            expected_margin = claims['pogq_beats_fltrust_at_45_percent']['expected_margin']

            status = 'PASS' if margin >= 0 else 'FAIL'
            results['checks'].append({
                'name': 'PoGQ > FLTrust @ 45%',
                'status': status,
                'expected_margin': expected_margin,
                'actual_margin': margin
            })
            if status == 'FAIL':
                results['status'] = 'FAIL'

    if 'pogq_99_at_10_percent' in claims:
        if '10%' in actual['data']:
            pogq_rate = actual['data']['10%']['pogq']['detection_rate_mean']
            min_value = claims['pogq_99_at_10_percent']['min_value']

            status = 'PASS' if pogq_rate >= min_value else 'WARN'
            results['checks'].append({
                'name': 'PoGQ ~99% @ 10%',
                'status': status,
                'min_value': min_value,
                'actual': pogq_rate
            })
            if status == 'WARN':
                results['warnings'].append(
                    f"PoGQ at 10%: expected >= {min_value*100:.0f}%, got {pogq_rate*100:.1f}%"
                )

    return results


def validate_table2(actual: Dict, expected: Dict) -> Dict:
    """Validate Table 2 (Latency) results."""
    results = {
        'name': 'Table 2: Aggregation Latency',
        'status': 'PASS',
        'checks': [],
        'warnings': []
    }

    if 'data' not in actual:
        results['status'] = 'FAIL'
        results['checks'].append({
            'name': 'Data presence',
            'status': 'FAIL',
            'message': 'No data found in actual results'
        })
        return results

    expected_data = expected.get('data', {})

    for method in ['pogq', 'fltrust', 'krum', 'fedavg']:
        if method not in actual['data']:
            continue
        if method not in expected_data:
            continue

        actual_median = actual['data'][method]['median_ms']
        expected_median = expected_data[method]['median_ms']
        tolerance_factor = expected_data[method].get('tolerance_factor', 0.20)

        # Allow for hardware differences - actual can be slower
        lower_bound = expected_median * (1 - tolerance_factor)
        upper_bound = expected_median * (1 + tolerance_factor * 2)  # More lenient for slower hardware

        status = 'PASS' if lower_bound <= actual_median <= upper_bound else 'WARN'

        if status == 'WARN':
            if actual_median > upper_bound:
                results['warnings'].append(
                    f"{method}: {actual_median:.3f}ms slower than expected "
                    f"({expected_median:.3f}ms, +{(actual_median/expected_median-1)*100:.0f}%)"
                )
            else:
                results['warnings'].append(
                    f"{method}: {actual_median:.3f}ms faster than expected "
                    f"({expected_median:.3f}ms, {(actual_median/expected_median-1)*100:.0f}%)"
                )

        results['checks'].append({
            'name': f'{method} median latency',
            'status': status,
            'expected': expected_median,
            'actual': actual_median,
            'tolerance_factor': tolerance_factor
        })

    # Validate key claims
    claims = expected.get('key_claims', {})

    if 'pogq_sub_millisecond' in claims:
        if 'pogq' in actual['data']:
            pogq_median = actual['data']['pogq']['median_ms']
            max_value = claims['pogq_sub_millisecond']['max_value_ms']

            status = 'PASS' if pogq_median < max_value else 'WARN'
            results['checks'].append({
                'name': 'PoGQ sub-millisecond',
                'status': status,
                'max_value': max_value,
                'actual': pogq_median
            })
            if status == 'WARN':
                results['status'] = 'WARN'
                results['warnings'].append(
                    f"PoGQ not sub-millisecond: {pogq_median:.3f}ms > 1.0ms"
                )

    if 'pogq_faster_than_krum' in claims:
        if 'pogq' in actual['data'] and 'krum' in actual['data']:
            pogq_median = actual['data']['pogq']['median_ms']
            krum_median = actual['data']['krum']['median_ms']
            actual_speedup = krum_median / pogq_median
            min_speedup = claims['pogq_faster_than_krum']['min_speedup']

            status = 'PASS' if actual_speedup >= min_speedup else 'WARN'
            results['checks'].append({
                'name': f'PoGQ {min_speedup}x faster than Krum',
                'status': status,
                'min_speedup': min_speedup,
                'actual_speedup': actual_speedup
            })

    return results


def validate_all(actual_results: Dict = None,
                 results_dir: Path = None,
                 expected_dir: Path = None) -> Dict:
    """
    Validate all results against expected values.

    Args:
        actual_results: Pre-loaded results dict (optional)
        results_dir: Path to results directory (used if actual_results not provided)
        expected_dir: Path to expected results directory

    Returns:
        Dictionary with validation results
    """
    if expected_dir is None:
        expected_dir = Path(__file__).parent / 'expected_results'

    if actual_results is None:
        if results_dir is None:
            results_dir = Path(__file__).parent / 'output' / 'results'
        actual_results = load_actual_results(results_dir)

    expected = load_expected_results(expected_dir)

    validation = {
        'overall_status': 'PASS',
        'experiments': []
    }

    # Validate Table 1
    if 'table1' in actual_results and 'table1' in expected:
        table1_result = validate_table1(actual_results['table1'], expected['table1'])
        validation['experiments'].append(table1_result)
        if table1_result['status'] == 'FAIL':
            validation['overall_status'] = 'FAIL'
        elif table1_result['status'] == 'WARN' and validation['overall_status'] == 'PASS':
            validation['overall_status'] = 'WARN'

    # Validate Table 2
    if 'table2' in actual_results and 'table2' in expected:
        table2_result = validate_table2(actual_results['table2'], expected['table2'])
        validation['experiments'].append(table2_result)
        if table2_result['status'] == 'FAIL':
            validation['overall_status'] = 'FAIL'
        elif table2_result['status'] == 'WARN' and validation['overall_status'] == 'PASS':
            validation['overall_status'] = 'WARN'

    return validation


def print_validation_report(validation: Dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    status_symbols = {'PASS': '[OK]', 'WARN': '[!!]', 'FAIL': '[XX]'}

    for exp in validation['experiments']:
        status = status_symbols.get(exp['status'], '[??]')
        print(f"\n{status} {exp['name']}")
        print("-" * 40)

        # Print checks
        for check in exp['checks']:
            check_status = status_symbols.get(check['status'], '[??]')
            if 'expected' in check and 'actual' in check:
                if isinstance(check['expected'], float):
                    print(f"  {check_status} {check['name']}: "
                          f"expected {check['expected']:.3f}, got {check['actual']:.3f}")
                else:
                    print(f"  {check_status} {check['name']}: "
                          f"expected {check['expected']}, got {check['actual']}")
            else:
                print(f"  {check_status} {check['name']}")

        # Print warnings
        if exp.get('warnings'):
            print("\n  Warnings:")
            for warning in exp['warnings']:
                print(f"    - {warning}")

    print("\n" + "=" * 60)
    overall = validation['overall_status']
    overall_symbol = status_symbols.get(overall, '[??]')
    print(f"OVERALL: {overall_symbol} {overall}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Validate Mycelix paper reproducibility results'
    )
    parser.add_argument('--results', type=str, default='output/results',
                       help='Path to results directory')
    parser.add_argument('--expected', type=str, default='expected_results',
                       help='Path to expected results directory')
    parser.add_argument('--json', action='store_true',
                       help='Output validation as JSON')
    args = parser.parse_args()

    results_dir = Path(__file__).parent / args.results
    expected_dir = Path(__file__).parent / args.expected

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run experiments first: python run_all.py")
        return 1

    validation = validate_all(
        results_dir=results_dir,
        expected_dir=expected_dir
    )

    if args.json:
        print(json.dumps(validation, indent=2))
    else:
        print_validation_report(validation)

    # Return exit code based on validation status
    if validation['overall_status'] == 'PASS':
        return 0
    elif validation['overall_status'] == 'WARN':
        return 0  # Warnings are acceptable
    else:
        return 1


if __name__ == '__main__':
    exit(main())
