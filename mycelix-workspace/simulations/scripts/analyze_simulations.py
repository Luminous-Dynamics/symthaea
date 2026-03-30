#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Publication-ready figures from simulation sweep CSV data.

Usage:
    python analyze_simulations.py --sweep-dir sweep-results/ --output-dir figures/
    python analyze_simulations.py --sweep-dir sweep-results/ --sim commons
    python analyze_simulations.py --sweep-dir sweep-results/ --sim macro
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# STYLE (matching symthaea/scripts/create_scaling_figures.py)
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'legend.fontsize': 10,
})

COLORS = {
    'consciousness': '#3498DB',
    'flat': '#27AE60',
    'plutocratic': '#E74C3C',
    'random': '#95a5a6',
    'nogovern': '#2C3E50',
}

MODE_LABELS = {
    'consciousness': 'Consciousness-Gated',
    'flat': 'Flat (1-person-1-vote)',
    'plutocratic': 'Plutocratic (extraction-weighted)',
    'random': 'Random Exclusion',
    'nogovern': 'No Governance',
}

# ============================================================================
# CSV UTILITIES
# ============================================================================

def read_csv(path):
    """Read CSV into list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def col_floats(rows, col):
    """Extract a column as float array."""
    return np.array([float(r[col]) for r in rows])


def col_ints(rows, col):
    """Extract a column as int array."""
    return np.array([int(r[col]) for r in rows])


def save_fig(fig, output_dir, name):
    """Save figure as PNG + PDF."""
    fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}.png + .pdf')


# ============================================================================
# STATE-SPACE FIGURES
# ============================================================================

def plot_state_space(sweep_dir, output_dir):
    ss_dir = sweep_dir / 'state-space'
    if not ss_dir.exists():
        print('  State-space sweep data not found, skipping.')
        return

    # --- Single heatmap (integration vs binding) ---
    hm_file = ss_dir / 'heatmap_binding_integration.csv'
    if not hm_file.exists():
        hm_file = ss_dir / 'heatmap_integration_binding.csv'
    if hm_file.exists():
        rows = read_csv(hm_file)
        # Determine axes from header
        cols = list(rows[0].keys())
        ax_a, ax_b = cols[0], cols[1]

        a_vals = sorted(set(float(r[ax_a]) for r in rows))
        b_vals = sorted(set(float(r[ax_b]) for r in rows))
        grid = np.zeros((len(b_vals), len(a_vals)))

        for r in rows:
            ai = a_vals.index(float(r[ax_a]))
            bi = b_vals.index(float(r[ax_b]))
            grid[bi, ai] = float(r['consciousness'])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid, origin='lower', aspect='auto',
                       extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]],
                       cmap='viridis')
        ax.set_xlabel(ax_a.replace('_', ' ').title())
        ax.set_ylabel(ax_b.replace('_', ' ').title())
        ax.set_title('Consciousness Level: 2D Parameter Sweep')
        plt.colorbar(im, ax=ax, label='C(t)')

        # Add contour lines
        X, Y = np.meshgrid(a_vals, b_vals)
        ax.contour(X, Y, grid, levels=[0.01, 0.05, 0.1, 0.2, 0.3],
                   colors='white', linewidths=0.8, linestyles='dashed')

        save_fig(fig, output_dir, 'fig_statespace_heatmap')

    # --- 7x7 pairwise grid ---
    components = ['attention', 'binding', 'efficacy', 'integration',
                  'knowledge', 'recursion', 'workspace']
    n = len(components)
    heatmaps_found = 0

    fig, axes = plt.subplots(n, n, figsize=(16, 16))
    fig.suptitle('Consciousness State-Space: All Pairwise Component Sweeps', fontsize=14, y=0.92)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])

            if i == j:
                # Diagonal: single-component sweep from transition data
                trans_file = ss_dir / 'transition.csv'
                if trans_file.exists():
                    rows = read_csv(trans_file)
                    comp_rows = [r for r in rows if r['component'] == components[i]]
                    if comp_rows:
                        vals = col_floats(comp_rows, 'value')
                        cons = col_floats(comp_rows, 'consciousness')
                        ax.plot(vals, cons, color=COLORS['consciousness'], linewidth=1.5)
                        ax.set_xlim(0, 1)
                ax.set_title(components[i][:4], fontsize=8)
            elif j > i:
                # Upper triangle: heatmap
                a, b = components[i], components[j]
                if a > b:
                    a, b = b, a
                hm_file = ss_dir / f'heatmap_{a}_{b}.csv'
                if hm_file.exists():
                    rows = read_csv(hm_file)
                    cols = list(rows[0].keys())
                    ax_a, ax_b = cols[0], cols[1]
                    a_vals = sorted(set(float(r[ax_a]) for r in rows))
                    b_vals = sorted(set(float(r[ax_b]) for r in rows))
                    grid = np.zeros((len(b_vals), len(a_vals)))
                    for r in rows:
                        ai = a_vals.index(float(r[ax_a]))
                        bi = b_vals.index(float(r[ax_b]))
                        grid[bi, ai] = float(r['consciousness'])
                    ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis',
                              vmin=0, vmax=0.5)
                    heatmaps_found += 1
            else:
                # Lower triangle: blank
                ax.axis('off')

    if heatmaps_found > 0:
        save_fig(fig, output_dir, 'fig_statespace_grid')
    else:
        plt.close(fig)
        print('  No pairwise heatmaps found, skipping grid.')


# ============================================================================
# MACRO ECONOMY FIGURES
# ============================================================================

def plot_macro_economy(sweep_dir, output_dir):
    macro_dir = sweep_dir / 'macro-economy'
    if not macro_dir.exists():
        print('  Macro economy sweep data not found, skipping.')
        return

    # Find a default single-run CSV
    default = macro_dir / 'inactive_1.0_seed_42.csv'
    if not default.exists():
        csvs = sorted(macro_dir.glob('*.csv'))
        if not csvs:
            print('  No macro economy CSVs found.')
            return
        default = csvs[0]

    rows = read_csv(default)
    days = col_ints(rows, 'day')

    # --- SAP Gini over time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, col_floats(rows, 'sap_gini'), color=COLORS['consciousness'], linewidth=1.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('SAP Wealth Inequality Over Time')
    ax.set_ylim(0, 1)
    save_fig(fig, output_dir, 'fig_macro_gini')

    # --- TEND utilization ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, col_floats(rows, 'tend_utilization'), color='#E94F37', linewidth=1.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('TEND Utilization')
    ax.set_title('Mutual Credit Utilization Over Time')
    save_fig(fig, output_dir, 'fig_macro_tend')

    # --- Tier distribution stacked area ---
    fig, ax = plt.subplots(figsize=(12, 6))
    tier_names = ['Observer', 'Participant', 'Citizen', 'Steward', 'Guardian']
    tier_cols = ['tier_observer', 'tier_participant', 'tier_citizen', 'tier_steward', 'tier_guardian']
    tier_colors = ['#95a5a6', '#3498DB', '#27AE60', '#E94F37', '#8E44AD']
    tier_data = [col_ints(rows, c).astype(float) for c in tier_cols]
    ax.stackplot(days, *tier_data, labels=tier_names, colors=tier_colors, alpha=0.8)
    ax.set_xlabel('Day')
    ax.set_ylabel('Agent Count')
    ax.set_title('Consciousness Tier Distribution Over Time')
    ax.legend(loc='upper right')
    save_fig(fig, output_dir, 'fig_macro_tiers')

    # --- Sweep comparison: final Gini by inactive rate ---
    inactive_rates = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    seeds = [42, 123, 789, 1337, 2024, 3141, 4242, 5555, 6789, 9876]
    rate_means = []
    rate_stds = []
    valid_rates = []

    for rate in inactive_rates:
        finals = []
        for seed in seeds:
            f = macro_dir / f'inactive_{rate}_seed_{seed}.csv'
            if f.exists():
                r = read_csv(f)
                if r:
                    finals.append(float(r[-1]['sap_gini']))
        if finals:
            valid_rates.append(rate)
            rate_means.append(np.mean(finals))
            rate_stds.append(np.std(finals))

    if valid_rates:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(valid_rates)), rate_means, yerr=rate_stds,
               color=COLORS['consciousness'], alpha=0.8, capsize=5)
        ax.set_xticks(range(len(valid_rates)))
        ax.set_xticklabels([f'{r:.1f}x' for r in valid_rates])
        ax.set_xlabel('Inactivity Rate Multiplier')
        ax.set_ylabel('Final SAP Gini')
        ax.set_title('Wealth Inequality vs Agent Inactivity')
        save_fig(fig, output_dir, 'fig_macro_sweep')

    # --- Demurrage rate sweep ---
    demurrage_rates = [0.01, 0.02, 0.03, 0.04, 0.05]
    seeds = [42, 123, 789, 1337, 2024, 3141, 4242, 5555, 6789, 9876]
    d_means, d_stds, d_valid = [], [], []

    for rate in demurrage_rates:
        finals = []
        for seed in seeds:
            f = macro_dir / f'demurrage_{rate}_seed_{seed}.csv'
            if f.exists():
                r = read_csv(f)
                if r:
                    finals.append(float(r[-1]['sap_velocity']))
        if finals:
            d_valid.append(rate)
            d_means.append(np.mean(finals))
            d_stds.append(np.std(finals))

    if d_valid:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(d_valid, d_means, yerr=d_stds, color=COLORS['consciousness'],
                    linewidth=2, capsize=5, marker='o', markersize=6)
        ax.set_xlabel('Annual Demurrage Rate')
        ax.set_ylabel('Final SAP Velocity (tx/agent/day)')
        ax.set_title('SAP Velocity vs Demurrage Rate')
        save_fig(fig, output_dir, 'fig_macro_demurrage')

    # --- Jubilee timing sweep ---
    jubilee_years = [0, 2, 4, 8]
    j_means, j_stds, j_valid = [], [], []

    for years in jubilee_years:
        finals = []
        for seed in seeds:
            f = macro_dir / f'jubilee_{years}y_seed_{seed}.csv'
            if f.exists():
                r = read_csv(f)
                if r:
                    finals.append(float(r[-1]['mycel_gini']))
        if finals:
            j_valid.append(years)
            j_means.append(np.mean(finals))
            j_stds.append(np.std(finals))

    if j_valid:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ['Never' if y == 0 else f'{y}yr' for y in j_valid]
        ax.bar(range(len(j_valid)), j_means, yerr=j_stds,
               color='#8E44AD', alpha=0.8, capsize=5)
        ax.set_xticks(range(len(j_valid)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Jubilee Cycle')
        ax.set_ylabel('Final MYCEL Gini')
        ax.set_title('Reputation Inequality vs Jubilee Frequency (4-year sim)')
        save_fig(fig, output_dir, 'fig_macro_jubilee')


# ============================================================================
# COMMONS RESOURCE FIGURES
# ============================================================================

def plot_commons_resource(sweep_dir, output_dir):
    commons_dir = sweep_dir / 'commons-resource'
    if not commons_dir.exists():
        print('  Commons resource sweep data not found, skipping.')
        return

    # Find a default single-run CSV
    default = commons_dir / 'consciousness_def0.3_seed42.csv'
    if not default.exists():
        csvs = sorted(commons_dir.glob('consciousness_*.csv'))
        if not csvs:
            print('  No commons CSVs found.')
            return
        default = csvs[0]

    rows = read_csv(default)
    days = col_ints(rows, 'day')

    # --- Resource stock over time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, col_floats(rows, 'avg_stock'), color=COLORS['consciousness'], linewidth=1.5,
            label='Avg Stock Fraction')
    ax.plot(days, col_floats(rows, 'min_stock'), color=COLORS['plutocratic'], linewidth=1,
            alpha=0.6, label='Min Stock Fraction')
    ax.set_xlabel('Day')
    ax.set_ylabel('Stock Fraction')
    ax.set_title('Resource Stock Over Time (Consciousness-Gated)')
    ax.legend()
    ax.set_ylim(0, 1)
    save_fig(fig, output_dir, 'fig_commons_stock')

    # --- Governance ON vs OFF ---
    gov_on = commons_dir / 'consciousness_def0.3_seed42.csv'
    gov_off = commons_dir / 'nogovern_def0.3_seed42.csv'
    if gov_on.exists() and gov_off.exists():
        rows_on = read_csv(gov_on)
        rows_off = read_csv(gov_off)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        days_on = col_ints(rows_on, 'day')
        days_off = col_ints(rows_off, 'day')

        ax1.plot(days_on, col_floats(rows_on, 'avg_stock'), color=COLORS['consciousness'], linewidth=1.5)
        ax1.set_title('Governance ON (Consciousness-Gated)')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Avg Stock Fraction')
        ax1.set_ylim(0, 1)

        ax2.plot(days_off, col_floats(rows_off, 'avg_stock'), color=COLORS['nogovern'], linewidth=1.5)
        ax2.set_title('Governance OFF (No Governance)')
        ax2.set_xlabel('Day')
        ax2.set_ylim(0, 1)

        fig.suptitle('Resource Sustainability: Governance Impact (30% Defectors)', fontsize=13)
        fig.tight_layout()
        save_fig(fig, output_dir, 'fig_commons_governance')

    # --- Cooperation index ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, col_floats(rows, 'cooperation_index'), color=COLORS['flat'], linewidth=1.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Cooperation Index')
    ax.set_title('Cooperation Index Over Time')
    ax.set_ylim(0, 1)
    save_fig(fig, output_dir, 'fig_commons_cooperation')

    # --- HHI stewardship ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, col_floats(rows, 'hhi_stewardship'), color='#E94F37', linewidth=1.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('HHI (Stewardship Concentration)')
    ax.set_title('Governance Capture Risk Over Time')
    save_fig(fig, output_dir, 'fig_commons_hhi')

    # --- KEY FIGURE: Voting mode comparison ---
    defector_pcts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    seeds = [42, 123, 789, 1337, 2024, 3141, 4242, 5555, 6789, 9876]
    modes = ['consciousness', 'flat', 'plutocratic', 'random', 'nogovern']

    fig, ax = plt.subplots(figsize=(10, 6))

    for mode in modes:
        means = []
        stds = []
        valid_pcts = []

        for pct in defector_pcts:
            finals = []
            for seed in seeds:
                f = commons_dir / f'{mode}_def{pct}_seed{seed}.csv'
                if f.exists():
                    r = read_csv(f)
                    if r:
                        finals.append(float(r[-1]['sustainability_index']))
            if finals:
                valid_pcts.append(pct)
                means.append(np.mean(finals))
                stds.append(np.std(finals))

        if valid_pcts:
            color = COLORS.get(mode, '#333333')
            label = MODE_LABELS.get(mode, mode)
            ls = '--' if mode == 'nogovern' else '-'
            ax.errorbar(valid_pcts, means, yerr=stds, label=label,
                        color=color, linewidth=2, linestyle=ls,
                        capsize=4, marker='o', markersize=5)

    ax.set_xlabel('Defector Fraction')
    ax.set_ylabel('Final Sustainability Index')
    ax.set_title('Governance Mode Comparison: Sustainability vs Defector Rate')
    ax.legend(loc='lower left')
    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0, 1)
    save_fig(fig, output_dir, 'fig_commons_voting_comparison')

    # --- Cross-sim: consciousness degradation impact ---
    degradation_rates = [0.0, 0.001, 0.002, 0.005, 0.01]
    seeds = [42, 123, 789, 1337, 2024, 3141, 4242, 5555, 6789, 9876]
    deg_means, deg_stds, deg_valid = [], [], []

    for rate in degradation_rates:
        finals = []
        for seed in seeds:
            f = commons_dir / f'degrade_{rate}_seed{seed}.csv'
            if f.exists():
                r = read_csv(f)
                if r:
                    finals.append(float(r[-1]['sustainability_index']))
        if finals:
            deg_valid.append(rate)
            deg_means.append(np.mean(finals))
            deg_stds.append(np.std(finals))

    if len(deg_valid) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(deg_valid, deg_means, yerr=deg_stds, color='#E94F37',
                    linewidth=2, capsize=5, marker='s', markersize=6)
        ax.set_xlabel('Consciousness Degradation Rate (per day)')
        ax.set_ylabel('Final Sustainability Index')
        ax.set_title('Cross-Simulation: Substrate Degradation Impact on Commons')
        ax.set_ylim(0, 1)
        save_fig(fig, output_dir, 'fig_commons_degradation')


# ============================================================================
# SWARM FIGURES
# ============================================================================

def plot_swarm(sweep_dir, output_dir, swarm_csv=None, scaling_csv=None):
    # Look for swarm CSVs in sweep dir or explicit paths
    if swarm_csv and Path(swarm_csv).exists():
        swarm_path = Path(swarm_csv)
    else:
        candidates = list((sweep_dir / 'swarm').glob('*.csv')) if (sweep_dir / 'swarm').exists() else []
        if not candidates:
            print('  No swarm data found, skipping.')
            return
        swarm_path = candidates[0]

    rows = read_csv(swarm_path)
    ticks = col_ints(rows, 'tick')

    # --- Emergence ratio over ticks ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ticks, col_floats(rows, 'emergence_ratio'), color=COLORS['consciousness'], linewidth=1.5)
    ax.set_xlabel('Tick')
    ax.set_ylabel('Emergence Ratio')
    ax.set_title('Collective Consciousness Emergence Over Time')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Emergence = 1.0')
    ax.legend()
    save_fig(fig, output_dir, 'fig_swarm_emergence')

    # --- Scaling curve ---
    if scaling_csv and Path(scaling_csv).exists():
        srows = read_csv(Path(scaling_csv))
        n_agents = col_ints(srows, 'n_agents')
        emergence = col_floats(srows, 'emergence_ratio')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(n_agents, emergence, 'o-', color=COLORS['consciousness'], linewidth=2, markersize=8)
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Emergence Ratio')
        ax.set_title('Consciousness Emergence vs Swarm Size')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        save_fig(fig, output_dir, 'fig_swarm_scaling')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Simulation analysis & visualization')
    parser.add_argument('--sweep-dir', default='sweep-results', help='Sweep results directory')
    parser.add_argument('--output-dir', default='figures', help='Output directory for figures')
    parser.add_argument('--sim', default='all', choices=['all', 'state-space', 'macro', 'commons', 'swarm'],
                        help='Which simulation to analyze')
    parser.add_argument('--swarm-csv', default=None, help='Path to swarm single-run CSV')
    parser.add_argument('--scaling-csv', default=None, help='Path to swarm scaling CSV')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Sweep dir: {sweep_dir}')
    print(f'Output dir: {output_dir}')
    print()

    if args.sim in ('all', 'state-space'):
        print('=== State-Space Figures ===')
        plot_state_space(sweep_dir, output_dir)
        print()

    if args.sim in ('all', 'macro'):
        print('=== Macro Economy Figures ===')
        plot_macro_economy(sweep_dir, output_dir)
        print()

    if args.sim in ('all', 'commons'):
        print('=== Commons Resource Figures ===')
        plot_commons_resource(sweep_dir, output_dir)
        print()

    if args.sim in ('all', 'swarm'):
        print('=== Swarm Figures ===')
        plot_swarm(sweep_dir, output_dir, args.swarm_csv, args.scaling_csv)
        print()

    print('Done.')


if __name__ == '__main__':
    main()
