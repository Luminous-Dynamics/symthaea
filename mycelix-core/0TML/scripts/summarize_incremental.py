#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Incremental Summary: Rolling Table-II Preview
==============================================

Reads all completed experiment artifacts and generates a running summary
of AUROC, TPR, FPR across the experimental matrix.

Usage:
    python scripts/summarize_incremental.py
    python scripts/summarize_incremental.py | column -t  # Pretty table
    python scripts/summarize_incremental.py --csv > table_ii_preview.csv
"""

import json
import glob
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict
import argparse


def find_artifact_dirs():
    """Find all artifact directories in results/"""
    return sorted(glob.glob("/srv/luminous-dynamics/Mycelix-Core/0TML/results/artifacts_*/"))


def parse_experiment_metadata(artifact_dir):
    """Extract metadata from artifact directory or main result file"""
    # Try to find main result JSON
    result_files = [f for f in os.listdir(artifact_dir) if f.endswith('.json') and 'per_bucket' not in f and 'bootstrap' not in f]

    if result_files:
        main_file = os.path.join(artifact_dir, result_files[0])
        try:
            with open(main_file, 'r') as f:
                data = json.load(f)
                return data.get('meta', {}), data.get('metrics', {})
        except:
            pass

    # Parse from directory name: artifacts_YYYYMMDD_HHMMSS_{dataset}_{attack}_{defense}_seed{seed}
    dir_name = os.path.basename(artifact_dir.rstrip('/'))
    parts = dir_name.split('_')

    # Basic parsing (may need adjustment based on actual naming)
    meta = {
        'artifact_dir': dir_name,
        'timestamp': '_'.join(parts[1:3]) if len(parts) > 3 else 'unknown'
    }

    return meta, {}


def load_detection_metrics(artifact_dir):
    """Load detection_metrics.json if available"""
    metrics_file = os.path.join(artifact_dir, 'detection_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}


def load_bootstrap_ci(artifact_dir):
    """Load bootstrap_ci.json if available"""
    ci_file = os.path.join(artifact_dir, 'bootstrap_ci.json')
    if os.path.exists(ci_file, 'r') as f:
        return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description='Generate incremental Table-II preview')
    parser.add_argument('--csv', action='store_true', help='Output CSV format')
    parser.add_argument('--verbose', action='store_true', help='Show additional metrics')
    args = parser.parse_args()

    artifact_dirs = find_artifact_dirs()

    if not artifact_dirs:
        print("No artifact directories found in results/", file=sys.stderr)
        return

    rows = []

    for artifact_dir in artifact_dirs:
        meta, metrics = parse_experiment_metadata(artifact_dir)
        detection = load_detection_metrics(artifact_dir)

        # Extract key metrics
        dataset = meta.get('dataset', 'unknown')
        non_iid = meta.get('non_iid', False)
        attack = meta.get('attack', 'unknown')
        defense = meta.get('defense', 'unknown')
        byz_ratio = meta.get('byzantine_ratio', meta.get('f', 0.0))
        seed = meta.get('seed', 0)

        auroc = detection.get('auroc', metrics.get('auroc'))
        tpr = detection.get('tpr_at_target_fpr', metrics.get('tpr'))
        fpr = detection.get('fpr_at_target_tpr', metrics.get('fpr'))

        rows.append({
            'dataset': dataset,
            'non_iid': non_iid,
            'attack': attack,
            'defense': defense,
            'f': byz_ratio,
            'seed': seed,
            'auroc': auroc,
            'tpr': tpr,
            'fpr': fpr,
            'artifact_dir': os.path.basename(artifact_dir.rstrip('/'))
        })

    # Output
    if args.csv:
        # CSV format
        print("dataset,non_iid,attack,defense,byzantine_ratio,seed,auroc,tpr,fpr,artifact_dir")
        for r in rows:
            print(f"{r['dataset']},{r['non_iid']},{r['attack']},{r['defense']},"
                  f"{r['f']},{r['seed']},{r['auroc']},{r['tpr']},{r['fpr']},{r['artifact_dir']}")
    else:
        # Tab-separated for `column -t`
        header = "dataset\tnon_iid\tattack\tdefense\tf\tseed\tauroc\ttpr\tfpr"
        if args.verbose:
            header += "\tartifact_dir"
        print(header)

        for r in rows:
            line = f"{r['dataset']}\t{r['non_iid']}\t{r['attack']}\t{r['defense']}\t"
            line += f"{r['f']}\t{r['seed']}\t{r['auroc']}\t{r['tpr']}\t{r['fpr']}"
            if args.verbose:
                line += f"\t{r['artifact_dir']}"
            print(line)

    # Summary stats to stderr
    print(f"\n[Summary: {len(rows)} experiments found]", file=sys.stderr)
    if rows:
        datasets = set(r['dataset'] for r in rows)
        attacks = set(r['attack'] for r in rows)
        defenses = set(r['defense'] for r in rows)
        print(f"  Datasets: {len(datasets)} ({', '.join(sorted(datasets))})", file=sys.stderr)
        print(f"  Attacks: {len(attacks)} ({', '.join(sorted(attacks)[:3])}...)", file=sys.stderr)
        print(f"  Defenses: {len(defenses)} ({', '.join(sorted(defenses)[:3])}...)", file=sys.stderr)


if __name__ == '__main__':
    main()
