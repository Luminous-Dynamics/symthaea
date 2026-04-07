#!/usr/bin/env python3
"""Generate publication-quality figures from TSV data.

Run figure_data tests first:
  cargo test -p symthaea-nuclear --lib figure_data -- --nocapture --test-threads=1

Then:
  python3 plot_figures.py

Requires: matplotlib, numpy
"""

import os
import numpy as np

DATA_DIR = "/tmp/nuclear_figures"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib import cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — generating gnuplot scripts instead")


def load_tsv(name):
    path = os.path.join(DATA_DIR, name)
    with open(path) as f:
        header = f.readline().strip().split('\t')
        rows = []
        for line in f:
            vals = line.strip().split('\t')
            rows.append([float(v) for v in vals])
    return header, np.array(rows)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Superheavy Q_alpha Heatmap
# ═══════════════════════════════════════════════════════════════════════

def plot_superheavy_heatmap():
    header, data = load_tsv("superheavy_qalpha_heatmap.tsv")
    Z = data[:, 0].astype(int)
    N = data[:, 1].astype(int)
    Q = data[:, 3]

    z_unique = sorted(set(Z))
    n_unique = sorted(set(N))

    grid = np.full((len(z_unique), len(n_unique)), np.nan)
    z_map = {z: i for i, z in enumerate(z_unique)}
    n_map = {n: i for i, n in enumerate(n_unique)}

    for i in range(len(data)):
        grid[z_map[Z[i]], n_map[N[i]]] = Q[i]

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(grid, origin='lower', aspect='auto',
                       extent=[min(n_unique)-1, max(n_unique)+1,
                               min(z_unique)-1, max(z_unique)+1],
                       cmap='RdYlGn_r', vmin=4, vmax=16)
        ax.set_xlabel('Neutron Number N', fontsize=12)
        ax.set_ylabel('Proton Number Z', fontsize=12)
        ax.set_title(r'$Q_\alpha$ Landscape of Superheavy Elements (MeV)', fontsize=14)
        cbar = plt.colorbar(im, ax=ax, label=r'$Q_\alpha$ (MeV)')

        # Mark the most stable point
        min_idx = np.nanargmin(grid)
        min_zi, min_ni = np.unravel_index(min_idx, grid.shape)
        ax.plot(n_unique[min_ni], z_unique[min_zi], 'k*', markersize=15,
                label=f'Most stable: Z={z_unique[min_zi]}, N={n_unique[min_ni]}')
        ax.legend(loc='upper left', fontsize=10)

        # Mark N=184 line
        ax.axvline(x=184, color='white', linestyle='--', alpha=0.7, label='N=184')
        ax.text(184.5, max(z_unique)-0.5, 'N=184', color='white', fontsize=9)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, "fig1_superheavy_qalpha.pdf")
        plt.savefig(out, dpi=300)
        plt.savefig(out.replace('.pdf', '.png'), dpi=150)
        print(f"Saved: {out}")
    else:
        # Gnuplot script
        gp = os.path.join(OUT_DIR, "fig1_superheavy.gp")
        with open(gp, 'w') as f:
            f.write(f"""set terminal pdfcairo size 8,5 font "Times,12"
set output "{OUT_DIR}/fig1_superheavy_qalpha.pdf"
set xlabel "Neutron Number N"
set ylabel "Proton Number Z"
set title "Q_{{/Symbol a}} Landscape of Superheavy Elements (MeV)"
set palette defined (4 "dark-green", 8 "yellow", 12 "orange", 16 "red")
set cbrange [4:16]
set cblabel "Q_{{/Symbol a}} (MeV)"
set datafile separator "\\t"
set view map
set dgrid3d 9,21 splines
splot "{DATA_DIR}/superheavy_qalpha_heatmap.tsv" u 2:1:4 with pm3d notitle
""")
        print(f"Gnuplot script: {gp}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Shell Evolution (N=82 gap vs Z)
# ═══════════════════════════════════════════════════════════════════════

def plot_shell_evolution():
    header82, data82 = load_tsv("shell_evolution_n82.tsv")
    header50, data50 = load_tsv("shell_evolution_n50.tsv")

    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # N=82
        ax1.plot(data82[:, 0], data82[:, 4], 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Proton Number Z', fontsize=12)
        ax1.set_ylabel(r'$S_{2n}$ Shell Gap at N=82 (MeV)', fontsize=12)
        ax1.set_title('N=82 Shell Quenching', fontsize=14)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.annotate('Sn (Z=50)\n6.0 MeV', xy=(50, 6.04), fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='red'),
                     xytext=(42, 7.5), color='red')
        ax1.annotate('Nd (Z=60)\n3.9 MeV', xy=(60, 3.89), fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='red'),
                     xytext=(55, 2.5), color='red')
        ax1.grid(alpha=0.3)

        # N=50
        ax2.plot(data50[:, 0], data50[:, 4], 'rs-', linewidth=2, markersize=6)
        ax2.set_xlabel('Proton Number Z', fontsize=12)
        ax2.set_ylabel(r'$S_{2n}$ Shell Gap at N=50 (MeV)', fontsize=12)
        ax2.set_title('N=50 Shell Gap', fontsize=14)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, "fig2_shell_evolution.pdf")
        plt.savefig(out, dpi=300)
        plt.savefig(out.replace('.pdf', '.png'), dpi=150)
        print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Validation Splits (bar chart)
# ═══════════════════════════════════════════════════════════════════════

def plot_validation():
    # Hardcoded from our validation results
    labels = ['5-fold CV', 'Chain\nholdout', 'Proton\nholdout',
              'Frontier\n(N/Z≥1.4)', 'Superheavy\n(Z≥100)']
    rf_rms = np.array([0.665, 0.712, 0.751, 0.677, 0.760])
    dz_rms = np.array([0.668, 0.717, 0.730, 0.667, 0.755])

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, rf_rms, width, label='DZ10 + RF',
                       color='#2196F3', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, dz_rms, width, label='DZ10 only',
                       color='#FF9800', edgecolor='black', linewidth=0.5)

        ax.set_ylabel('RMS Error (MeV)', fontsize=12)
        ax.set_title('Extrapolation Stress Tests: RF Correction vs DZ10 Baseline', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Annotate the key finding
        ax.annotate('ML adds negligible\nextrapolation benefit',
                    xy=(3, 0.68), fontsize=9, ha='center', style='italic',
                    color='#666')

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, "fig3_validation_splits.pdf")
        plt.savefig(out, dpi=300)
        plt.savefig(out.replace('.pdf', '.png'), dpi=150)
        print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Wigner Energy
# ═══════════════════════════════════════════════════════════════════════

def plot_wigner():
    header, data = load_tsv("wigner_energy.tsv")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data[:, 0], data[:, 2], 'go-', linewidth=2, markersize=7)
        ax.set_xlabel('Proton Number Z (N=Z nuclei)', fontsize=12)
        ax.set_ylabel('Wigner Energy (MeV)', fontsize=12)
        ax.set_title('N=Z Wigner Energy (Isoscalar np Pairing)', fontsize=14)
        ax.grid(alpha=0.3)

        # Highlight Sn-100
        sn_idx = np.argmin(np.abs(data[:, 0] - 50))
        ax.annotate(f'Sn-100 (doubly magic)\n{data[sn_idx, 2]:.1f} MeV',
                    xy=(50, data[sn_idx, 2]),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    xytext=(42, 5.5), fontsize=10, color='red')

        plt.tight_layout()
        out = os.path.join(OUT_DIR, "fig4_wigner_energy.pdf")
        plt.savefig(out, dpi=300)
        plt.savefig(out.replace('.pdf', '.png'), dpi=150)
        print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: R-Process Abundance Pattern
# ═══════════════════════════════════════════════════════════════════════

def plot_rprocess():
    header, data = load_tsv("rprocess_abundances.tsv")
    A = data[:, 0].astype(int)
    Y = data[:, 1]

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(A, Y, 'b-', linewidth=1.5, alpha=0.8, label='DZ10+RF r-process')
        ax.semilogy(A, Y, 'b.', markersize=4, alpha=0.6)

        # Mark expected solar r-process peaks
        peak_positions = [80, 130, 195]
        peak_labels = ['A~80\n(N=50)', 'A~130\n(N=82)', 'A~195\n(N=126)']
        peak_colors = ['#e74c3c', '#2ecc71', '#9b59b6']
        for pos, label, color in zip(peak_positions, peak_labels, peak_colors):
            ax.axvline(x=pos, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
            ax.text(pos + 2, ax.get_ylim()[1] * 0.3, label,
                    color=color, fontsize=10, fontweight='bold',
                    verticalalignment='top')

        ax.set_xlabel('Mass Number A', fontsize=13)
        ax.set_ylabel('Abundance Y(A)', fontsize=13)
        ax.set_title('R-Process Nucleosynthesis: DZ10+RF Network Simulation\n'
                      r'($T_9 = 2.0$ GK, $n_n = 10^{24}$ cm$^{-3}$, '
                      r'$\tau_{exp} = 0.85$ s)', fontsize=13)
        ax.set_xlim(60, 320)
        ax.set_ylim(1e-8, 2)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(alpha=0.3, which='both')

        # Annotate the two main peaks
        # Find the peak in A~130 region
        mask_130 = (A >= 125) & (A <= 140)
        if np.any(mask_130):
            peak_a = A[mask_130][np.argmax(Y[mask_130])]
            peak_y = Y[mask_130].max()
            ax.annotate(f'A={peak_a}\nY={peak_y:.2e}',
                        xy=(peak_a, peak_y),
                        xytext=(peak_a + 15, peak_y * 2),
                        fontsize=9, color='#2ecc71',
                        arrowprops=dict(arrowstyle='->', color='#2ecc71'))

        # Find the peak in A~195-210 region (broadened third peak)
        mask_195 = (A >= 190) & (A <= 210)
        if np.any(mask_195):
            peak_a = A[mask_195][np.argmax(Y[mask_195])]
            peak_y = Y[mask_195].max()
            ax.annotate(f'A={peak_a}\nY={peak_y:.2e}',
                        xy=(peak_a, peak_y),
                        xytext=(peak_a + 15, peak_y * 5),
                        fontsize=9, color='#9b59b6',
                        arrowprops=dict(arrowstyle='->', color='#9b59b6'))

        plt.tight_layout()
        out = os.path.join(OUT_DIR, "fig5_rprocess_abundances.pdf")
        plt.savefig(out, dpi=300)
        plt.savefig(out.replace('.pdf', '.png'), dpi=150)
        print(f"Saved: {out}")
    else:
        gp = os.path.join(OUT_DIR, "fig5_rprocess.gp")
        with open(gp, 'w') as f:
            f.write(f"""set terminal pdfcairo size 10,6 font "Times,12"
set output "{OUT_DIR}/fig5_rprocess_abundances.pdf"
set xlabel "Mass Number A"
set ylabel "Abundance Y(A)"
set title "R-Process Nucleosynthesis: DZ10+RF Network Simulation"
set logscale y
set format y "10^{{%L}}"
set xrange [60:320]
set yrange [1e-8:2]
set datafile separator "\\t"
set grid
set arrow from 80,1e-8 to 80,2 nohead dt 2 lc rgb "red"
set arrow from 130,1e-8 to 130,2 nohead dt 2 lc rgb "green"
set arrow from 195,1e-8 to 195,2 nohead dt 2 lc rgb "purple"
plot "{DATA_DIR}/rprocess_abundances.tsv" u 1:2 with linespoints pt 7 ps 0.5 lw 1.5 title "DZ10+RF r-process"
""")
        print(f"Gnuplot script: {gp}")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print()
    plot_superheavy_heatmap()
    plot_shell_evolution()
    plot_validation()
    plot_wigner()
    plot_rprocess()
    print("\nDone. Figures saved to:", OUT_DIR)
