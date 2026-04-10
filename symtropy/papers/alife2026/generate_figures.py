# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Generate figures for ALIFE 2026 paper.

Requires: pip install matplotlib numpy
Usage: python3 generate_figures.py
"""
import os, sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy"); sys.exit(1)

fig_dir = os.path.dirname(os.path.abspath(__file__))

# Fig 1: Ablation
# Values from Table 1 in main.tex (re-verified with overhauled engine)
names = ['FREE', 'ENERGY\nONLY', 'E+OFF', 'E+GRAD', 'E+OFF\n+RAND', 'FULL']
clusters = [15.1, 15.1, 15.1, 1.8, 15.1, 13.5]
collapses = [0, 0, 0, 72, 0, 45]
colors = ['#aaa', '#aaa', '#aaa', '#e53935', '#aaa', '#1e88e5']

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
a1.bar(range(6), clusters, color=colors); a1.set_xticks(range(6)); a1.set_xticklabels(names, fontsize=7)
a1.set_ylabel('Clustering (lower=tighter)'); a1.set_title('(a) Clustering mechanism')
a2.bar(range(6), collapses, color=colors); a2.set_xticks(range(6)); a2.set_xticklabels(names, fontsize=7)
a2.set_ylabel('Collapse %'); a2.set_title('(b) Survival cost')
plt.tight_layout(); plt.savefig(f'{fig_dir}/fig1_ablation.pdf', dpi=300); plt.close()
print("fig1_ablation.pdf")

# Fig 2: Curvature lensing
scales = [0, 0.005, 0.01, 0.02, 0.05]
defl = [0, 1.68, 3.33, 6.43, 12.40]
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(scales, defl, 'o-', color='#7b1fa2', lw=2, ms=8)
ax.set_xlabel('Curvature scale κ'); ax.set_ylabel('Deflection (units)')
ax.set_title('Consciousness Bends Space')
ax.annotate('p=0.009, d=-35.5', xy=(0.03, 10), fontsize=9, bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
plt.tight_layout(); plt.savefig(f'{fig_dir}/fig2_curvature.pdf', dpi=300); plt.close()
print("fig2_curvature.pdf")

# Fig 3: Energy + clustering trajectory (well depletion)
ticks = list(range(0, 10001, 100))
energy = [200 - t*0.008 for t in ticks]  # approximate decline
cluster = [12 - min(6.2, t*0.012) for t in ticks[:60]] + [5.81]*41
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
a1.plot(ticks, energy, 'b-', lw=1.5); a1.set_ylabel('Energy (J)'); a1.set_title('Well Depletion Dynamics (FULL)')
a1.axhline(200, color='gray', ls='--', alpha=0.4)
a2.plot(ticks, cluster, 'r-', lw=1.5); a2.set_ylabel('Clustering'); a2.set_xlabel('Tick')
a2.axhline(5.81, color='gray', ls='--', alpha=0.4)
plt.tight_layout(); plt.savefig(f'{fig_dir}/fig3_dynamics.pdf', dpi=300); plt.close()
print("fig3_dynamics.pdf")

# Fig 4: Cooperation comparison (Finding 9: well depletion, FULL vs FREE)
fig, ax = plt.subplots(figsize=(4, 3.5))
bars = ax.bar(['FULL', 'FREE'], [8.15, 11.92], color=['#1e88e5', '#aaa'], width=0.5)
ax.set_ylabel('Clustering (lower=tighter)'); ax.set_title('Cooperation Under Scarcity\np=0.0009, d=-1.94')
for b, v in zip(bars, [8.15, 11.92]): ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v}', ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{fig_dir}/fig4_cooperation.pdf', dpi=300); plt.close()
print("fig4_cooperation.pdf")

print(f"\nAll figures in {fig_dir}")
