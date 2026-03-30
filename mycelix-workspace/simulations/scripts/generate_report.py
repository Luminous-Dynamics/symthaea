#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Generate a markdown summary report from simulation sweep results.

Reads all sweep CSVs and produces a structured report with key findings,
statistical summaries, and figure references.

Usage:
    python generate_report.py --sweep-dir sweep-results/ --figures-dir figures/ --output report.md
"""

import argparse
import csv
import os
from pathlib import Path
import numpy as np


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def final_value(rows, col):
    if not rows:
        return None
    return float(rows[-1].get(col, 0))


def sweep_stats(sweep_dir, prefix, param_values, metric, seeds):
    """Compute mean +/- std of final metric across seeds for each param value."""
    results = []
    for pv in param_values:
        finals = []
        for seed in seeds:
            f = sweep_dir / f'{prefix}{pv}_seed{seed}.csv' if '_seed' not in prefix else \
                sweep_dir / f'{prefix}{pv}_seed_{seed}.csv'
            # Try both naming conventions
            for pattern in [f'{prefix}{pv}_seed{seed}.csv', f'{prefix}{pv}_seed_{seed}.csv']:
                fp = sweep_dir / pattern
                if fp.exists():
                    r = read_csv(fp)
                    if r:
                        v = final_value(r, metric)
                        if v is not None:
                            finals.append(v)
                    break
        if finals:
            results.append((pv, np.mean(finals), np.std(finals), len(finals)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-dir', default='sweep-results')
    parser.add_argument('--figures-dir', default='figures')
    parser.add_argument('--output', default='SIMULATION_REPORT.md')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    figs = Path(args.figures_dir)
    seeds = [42, 123, 789, 1337, 2024, 3141, 4242, 5555, 6789, 9876]

    lines = []
    def w(s=''):
        lines.append(s)

    w('# Simulation Suite Report')
    w()
    w('Generated from parameter sweep results across 10 seeds.')
    w()

    # ── Commons Resource ──
    commons_dir = sweep_dir / 'commons-resource'
    if commons_dir.exists():
        w('## Commons Resource Sustainability')
        w()
        w('Tests Ostrom\'s design principles for commons governance with consciousness-gated access.')
        w('Calibrated against Spanish Huerta irrigation system (Ostrom 1990, Ch. 3).')
        w()
        w('### Model')
        w('- 100 agents, 10 resources, 365 days')
        w('- 10 agent strategies: Steward, Moderate, Cooperator, Altruist, Seasonal, StrategicCooperator, CaptureSeeker, Defector, FreeRider, Overextractor')
        w('- Sustainable yield extraction: quotas computed from regeneration budget')
        w('- Small-world network topology (4 nearest + 2 random neighbors)')
        w('- 5 resource types with agent specialization')
        w()

        # Voting mode comparison
        w('### Key Finding 1: Governance Mode Comparison')
        w()
        modes = ['consciousness', 'flat', 'plutocratic', 'random', 'nogovern']
        defpcts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        w('| Defector % | Consciousness | Flat | Plutocratic | Random | No Gov |')
        w('|-----------|--------------|------|-------------|--------|--------|')
        for dp in defpcts:
            row = f'| {dp:.0%} |'
            for mode in modes:
                stats = sweep_stats(commons_dir, f'{mode}_def{dp}_', [''], 'sustainability_index', seeds)
                if not stats:
                    # Try direct pattern
                    finals = []
                    for seed in seeds:
                        f = commons_dir / f'{mode}_def{dp}_seed{seed}.csv'
                        if f.exists():
                            r = read_csv(f)
                            if r:
                                finals.append(final_value(r, 'sustainability_index'))
                    if finals:
                        row += f' {np.mean(finals):.3f} +/- {np.std(finals):.3f} |'
                    else:
                        row += ' - |'
                else:
                    _, m, s, _ = stats[0]
                    row += f' {m:.3f} +/- {s:.3f} |'
            w(row)
        w()
        w('![Voting Comparison](figures/fig_commons_voting_comparison.png)')
        w()

        # Degradation
        w('### Key Finding 2: Cross-Simulation Consciousness Degradation')
        w()
        deg_rates = [0.0, 0.001, 0.002, 0.005, 0.01]
        for rate in deg_rates:
            finals = []
            for seed in seeds:
                f = commons_dir / f'degrade_{rate}_seed{seed}.csv'
                if f.exists():
                    r = read_csv(f)
                    if r:
                        finals.append(final_value(r, 'sustainability_index'))
            if finals:
                w(f'- Degradation {rate}/day: sustainability = {np.mean(finals):.3f} +/- {np.std(finals):.3f} (n={len(finals)})')
        w()
        w('![Degradation Impact](figures/fig_commons_degradation.png)')
        w()

    # ── Macro Economy ──
    macro_dir = sweep_dir / 'macro-economy'
    if macro_dir.exists():
        w('## Multi-Currency Macro Economy')
        w()
        w('Triple-currency system (SAP/TEND/MYCEL) with demurrage, counter-cyclical credit, and consciousness-gated tiers.')
        w()

        # Demurrage finding
        w('### Key Finding 3: Demurrage Rate Has No Significant Effect on Velocity')
        w()
        w('Within the constitutional bounds (1-5%), demurrage rate does not significantly')
        w('affect SAP transaction velocity. The exempt floor (1,000 SAP) dominates — most')
        w('agents hold near or below the floor, making the rate irrelevant for circulation.')
        w()
        for rate in [0.01, 0.02, 0.03, 0.04, 0.05]:
            finals = []
            for seed in seeds:
                f = macro_dir / f'demurrage_{rate}_seed_{seed}.csv'
                if f.exists():
                    r = read_csv(f)
                    if r:
                        finals.append(final_value(r, 'sap_velocity'))
            if finals:
                w(f'- Demurrage {rate:.0%}: velocity = {np.mean(finals):.4f} +/- {np.std(finals):.4f} tx/agent/day')
        w()
        w('**Implication**: The demurrage rate within constitutional bounds is a political parameter,')
        w('not an economic lever. The exempt floor is the binding constraint.')
        w()
        w('![Demurrage vs Velocity](figures/fig_macro_demurrage.png)')
        w()

        # Jubilee
        w('### Key Finding 4: Jubilee Frequency and Reputation Inequality')
        w()
        for years in [0, 2, 4, 8]:
            finals = []
            for seed in seeds:
                f = macro_dir / f'jubilee_{years}y_seed_{seed}.csv'
                if f.exists():
                    r = read_csv(f)
                    if r:
                        finals.append(final_value(r, 'mycel_gini'))
            if finals:
                label = 'Never' if years == 0 else f'{years}-year'
                w(f'- {label} jubilee: MYCEL Gini = {np.mean(finals):.4f} +/- {np.std(finals):.4f}')
        w()
        w('![Jubilee Impact](figures/fig_macro_jubilee.png)')
        w()

        # Inactivity
        w('### Key Finding 5: Inactivity and Wealth Concentration')
        w()
        for rate in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            finals = []
            for seed in seeds:
                f = macro_dir / f'inactive_{rate}_seed_{seed}.csv'
                if f.exists():
                    r = read_csv(f)
                    if r:
                        finals.append(final_value(r, 'sap_gini'))
            if finals:
                w(f'- Inactivity {rate:.1f}x: SAP Gini = {np.mean(finals):.4f} +/- {np.std(finals):.4f}')
        w()
        w('![Inactivity vs Gini](figures/fig_macro_sweep.png)')
        w()

    # ── Methodology ──
    w('## Methodology')
    w()
    w('- **Seeds**: 10 independent runs per configuration (42, 123, 789, 1337, 2024, 3141, 4242, 5555, 6789, 9876)')
    w('- **Error bars**: Mean +/- 1 standard deviation')
    w('- **Calibration**: Commons parameters calibrated against Ostrom (1990) Spanish Huerta system')
    w('- **Extraction model**: Sustainable yield — cooperative quotas derived from logistic regeneration')
    w('- **Network**: Watts-Strogatz small-world (k=4 nearest + 2 random long-range)')
    w('- **Voting modes**: Consciousness-gated (sigmoid weight), Flat (equal), Plutocratic (extraction-weighted), Random (probabilistic exclusion)')
    w()

    # ── File counts ──
    n_csvs = len(list(sweep_dir.rglob('*.csv'))) if sweep_dir.exists() else 0
    n_figs = len(list(figs.glob('*.png'))) if figs.exists() else 0
    w(f'## Data')
    w()
    w(f'- **{n_csvs}** sweep CSV files')
    w(f'- **{n_figs}** publication figures (PNG + PDF)')
    w(f'- **10** random seeds for statistical robustness')
    w()

    # Write output
    output = Path(args.output)
    output.write_text('\n'.join(lines))
    print(f'Report written to {output} ({len(lines)} lines)')


if __name__ == '__main__':
    main()
