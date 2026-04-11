# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate J/Phi paper figures from REAL simulation data.

Reads jphi_real_data.csv (output of cargo run --example jphi_convergence).
"""
import os, sys, csv

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy"); sys.exit(1)

fig_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(fig_dir, 'jphi_real_data.csv')

if not os.path.exists(data_file):
    print(f"Missing {data_file} — run: cargo run --example jphi_convergence --release")
    sys.exit(1)

# Load CSV
rows = []
with open(data_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            'condition': row['condition'],
            'seed': int(row['seed']),
            'tick': int(row['tick']),
            'phi': float(row['collective_phi']),
            'jphi': float(row['joules_per_phi']) if row['joules_per_phi'] != 'N/A' else float('nan'),
            'alive': int(row['alive']),
            'energy': float(row['avg_energy']),
            'clustering': float(row['clustering']),
        })

conditions = ['FULL', 'ENERGY_ONLY', 'FREE']
colors = {'FULL': '#1e88e5', 'ENERGY_ONLY': '#e53935', 'FREE': '#43a047'}
labels = {'FULL': 'FULL (thermo+FEP+offload)', 'ENERGY_ONLY': 'ENERGY_ONLY', 'FREE': 'FREE (control)'}

# ═══════════════════════════════════════════════════════════
# Fig 1: J/Phi convergence time series (real data, seed 42)
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 4.5))
for cond in conditions:
    cond_data = [r for r in rows if r['condition'] == cond and r['seed'] == 42]
    ticks = [r['tick'] for r in cond_data]
    jphi = [r['jphi'] for r in cond_data]
    # Clip extreme values for readability
    jphi_clipped = [min(j, 5000) for j in jphi]
    ax.plot(ticks, jphi_clipped, '-', color=colors[cond], lw=1.2, alpha=0.8, label=labels[cond])

ax.set_xlabel('Tick', fontsize=11)
ax.set_ylabel('J/Φ (Joules per unit integration change)', fontsize=11)
ax.set_title('J/Φ Convergence — Real Simulation Data (seed=42)', fontsize=13)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, max(r['tick'] for r in rows))
plt.tight_layout()
plt.savefig(f'{fig_dir}/fig1_jphi_convergence_real.pdf', dpi=300)
plt.close()
print("fig1_jphi_convergence_real.pdf")

# ═══════════════════════════════════════════════════════════
# Fig 2: Clustering comparison (real data, mean across seeds)
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(5, 4))
means = []
stds = []
for cond in conditions:
    # Get final tick clustering for each seed
    seeds = set(r['seed'] for r in rows if r['condition'] == cond)
    final_clusters = []
    for seed in seeds:
        seed_data = [r for r in rows if r['condition'] == cond and r['seed'] == seed]
        if seed_data:
            final_clusters.append(seed_data[-1]['clustering'])
    means.append(np.mean(final_clusters))
    stds.append(np.std(final_clusters))

cond_labels = ['FULL\n(cooperative)', 'ENERGY_ONLY\n(isolated)', 'FREE\n(unlimited)']
bars = ax.bar(cond_labels, means, yerr=stds, color=[colors[c] for c in conditions],
              width=0.5, capsize=5, error_kw={'lw': 1.5})
ax.set_ylabel('Final Clustering (lower = tighter)', fontsize=11)
ax.set_title('Cooperation Effect on Clustering\n(Real Data, 10 seeds)', fontsize=12)
for b, v in zip(bars, means):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f'{v:.2f}', ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(f'{fig_dir}/fig2_clustering_real.pdf', dpi=300)
plt.close()
print("fig2_clustering_real.pdf")

# ═══════════════════════════════════════════════════════════
# Fig 3: Energy trajectory (real data, all conditions)
# ═══════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for cond in conditions:
    cond_data = [r for r in rows if r['condition'] == cond and r['seed'] == 42]
    ticks = [r['tick'] for r in cond_data]
    energy = [r['energy'] for r in cond_data]
    phi = [r['phi'] for r in cond_data]
    ax1.plot(ticks, energy, '-', color=colors[cond], lw=1.2, label=labels[cond])
    ax2.plot(ticks, phi, '-', color=colors[cond], lw=1.2, label=labels[cond])

ax1.set_ylabel('Mean Energy (J)', fontsize=10)
ax1.set_title('Energy & Phi Trajectories (seed=42)', fontsize=12)
ax1.legend(fontsize=8)
ax2.set_ylabel('Collective Φ', fontsize=10)
ax2.set_xlabel('Tick', fontsize=10)
plt.tight_layout()
plt.savefig(f'{fig_dir}/fig3_energy_phi_real.pdf', dpi=300)
plt.close()
print("fig3_energy_phi_real.pdf")

# ═══════════════════════════════════════════════════════════
# Fig 4: Survival across seeds
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 4))
for cond in conditions:
    seeds = sorted(set(r['seed'] for r in rows if r['condition'] == cond))
    for seed in seeds:
        seed_data = [r for r in rows if r['condition'] == cond and r['seed'] == seed]
        ticks = [r['tick'] for r in seed_data]
        alive = [r['alive'] for r in seed_data]
        ax.plot(ticks, alive, '-', color=colors[cond], alpha=0.3, lw=0.8)
    # Mean line
    mean_alive = {}
    for r in rows:
        if r['condition'] == cond:
            mean_alive.setdefault(r['tick'], []).append(r['alive'])
    ticks_sorted = sorted(mean_alive.keys())
    means = [np.mean(mean_alive[t]) for t in ticks_sorted]
    ax.plot(ticks_sorted, means, '-', color=colors[cond], lw=2.5, label=f'{cond} (mean)')

ax.set_xlabel('Tick', fontsize=11)
ax.set_ylabel('Agents Alive', fontsize=11)
ax.set_title('Survival Across Seeds (thin=individual, thick=mean)', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{fig_dir}/fig4_survival_real.pdf', dpi=300)
plt.close()
print("fig4_survival_real.pdf")

print(f"\nAll real-data figures saved to {fig_dir}/")
print(f"Data: {len(rows)} rows from {data_file}")
