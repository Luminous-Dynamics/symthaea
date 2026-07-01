#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Create Scaling Figures

Generates publication-quality figures for the phenomenal discrimination research:
1. Full encoder scaling curve with optimal peak
2. Encoder vs decoder comparison
3. Angular separation mechanism
4. Summary figure

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy matplotlib])" \
        --run "python3 scripts/create_scaling_figures.py"
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

# Output directory
OUTPUT_DIR = Path("/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load all experimental results."""
    data_dir = Path("/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data")

    with open(data_dir / "smaller_models_scaling.json") as f:
        encoder_data = json.load(f)

    with open(data_dir / "gpt2_phenomenal_scaling.json") as f:
        decoder_data = json.load(f)

    with open(data_dir / "mechanistic_circuit_analysis.json") as f:
        mechanistic_data = json.load(f)

    return encoder_data, decoder_data, mechanistic_data


def figure1_encoder_scaling_curve(encoder_data):
    """Figure 1: Full encoder scaling curve showing non-monotonic pattern."""
    fig, ax = plt.subplots(figsize=(10, 6))

    results = encoder_data["results"]

    # Sort by parameters
    results_sorted = sorted(results, key=lambda x: x["n_params_millions"])

    params = [r["n_params_millions"] for r in results_sorted]
    fishers = [r["fisher_criterion"] for r in results_sorted]
    names = [r["display_name"] for r in results_sorted]

    # Find optimal
    best_idx = np.argmax(fishers)

    # Plot
    ax.plot(params, fishers, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='Fisher Criterion')

    # Highlight optimal
    ax.scatter([params[best_idx]], [fishers[best_idx]], s=200, c='#E94F37', zorder=5,
               label=f'Optimal: {names[best_idx]}')

    # Add trend regions
    ax.axvspan(0, 50, alpha=0.1, color='blue', label='Rising phase')
    ax.axvspan(80, 150, alpha=0.1, color='green', label='Optimal zone')
    ax.axvspan(200, 400, alpha=0.1, color='red', label='Declining phase')

    # Annotate key points
    ax.annotate(f'{names[best_idx]}\nF={fishers[best_idx]:.3f}',
                xy=(params[best_idx], fishers[best_idx]),
                xytext=(params[best_idx]+50, fishers[best_idx]+0.05),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel("Fisher's Criterion")
    ax.set_title('Non-Monotonic Scaling of Phenomenal Discrimination\n(Encoder Models)')
    ax.set_xscale('log')
    ax.legend(loc='lower left')
    ax.set_ylim(0.9, 1.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_encoder_scaling_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_encoder_scaling_curve.pdf', bbox_inches='tight')
    print(f"Saved: fig1_encoder_scaling_curve.png/pdf")
    plt.close()


def figure2_encoder_vs_decoder(encoder_data, decoder_data):
    """Figure 2: Comparison of encoder and decoder optimal sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Encoder data
    enc_results = sorted(encoder_data["results"], key=lambda x: x["n_params_millions"])
    enc_params = [r["n_params_millions"] for r in enc_results]
    enc_fishers = [r["fisher_criterion"] for r in enc_results]
    enc_best_idx = np.argmax(enc_fishers)

    # Decoder data
    dec_results = sorted(decoder_data["results"], key=lambda x: x["n_params_millions"])
    dec_params = [r["n_params_millions"] for r in dec_results]
    dec_fishers = [r["fisher_criterion"] for r in dec_results]
    dec_best_idx = np.argmax(dec_fishers)

    # Plot encoders
    ax1.plot(enc_params, enc_fishers, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax1.scatter([enc_params[enc_best_idx]], [enc_fishers[enc_best_idx]],
                s=200, c='#E94F37', zorder=5)
    ax1.axvline(enc_params[enc_best_idx], color='#E94F37', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Parameters (Millions)')
    ax1.set_ylabel("Fisher's Criterion")
    ax1.set_title(f'Encoders (BERT family)\nOptimal: ~{enc_params[enc_best_idx]:.0f}M')
    ax1.set_xscale('log')
    ax1.set_ylim(0.9, 1.25)

    # Plot decoders
    ax2.plot(dec_params, dec_fishers, 's-', color='#A23B72', linewidth=2, markersize=8)
    ax2.scatter([dec_params[dec_best_idx]], [dec_fishers[dec_best_idx]],
                s=200, c='#E94F37', zorder=5)
    ax2.axvline(dec_params[dec_best_idx], color='#E94F37', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Parameters (Millions)')
    ax2.set_ylabel("Fisher's Criterion")
    ax2.set_title(f'Decoders (GPT-2 family)\nOptimal: ~{dec_params[dec_best_idx]:.0f}M')
    ax2.set_xscale('log')
    ax2.set_ylim(0.65, 0.90)

    fig.suptitle('Architecture-Dependent Optimal Sizes for Phenomenal Discrimination',
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_encoder_vs_decoder.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_encoder_vs_decoder.pdf', bbox_inches='tight')
    print(f"Saved: fig2_encoder_vs_decoder.png/pdf")
    plt.close()


def figure3_angular_separation(mechanistic_data):
    """Figure 3: Angular separation mechanism."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    results = mechanistic_data["results"]

    # Extract data for BERT and RoBERTa
    bert_base = next(r for r in results if r["display_name"] == "BERT-base")
    bert_large = next(r for r in results if r["display_name"] == "BERT-large")
    roberta_base = next(r for r in results if r["display_name"] == "RoBERTa-base")
    roberta_large = next(r for r in results if r["display_name"] == "RoBERTa-large")

    # Angular separation comparison
    models = ['BERT-base', 'BERT-large', 'RoBERTa-base', 'RoBERTa-large']
    angular_seps = [
        bert_base["geometry"]["angular_separation"],
        bert_large["geometry"]["angular_separation"],
        roberta_base["geometry"]["angular_separation"],
        roberta_large["geometry"]["angular_separation"]
    ]
    colors = ['#2E86AB', '#1a5276', '#A23B72', '#6c1d45']

    bars = ax1.bar(models, angular_seps, color=colors)
    ax1.set_ylabel('Angular Separation')
    ax1.set_title('Angular Separation Between\nPhenomenal & Functional Centroids')
    ax1.set_ylim(0, 0.30)

    # Add percentage change annotations
    ax1.annotate('-40%', xy=(1, angular_seps[1]), xytext=(1, angular_seps[1]+0.03),
                ha='center', fontsize=10, color='red')

    # Isotropy comparison
    isotropies = [
        bert_base["geometry"]["isotropy"],
        bert_large["geometry"]["isotropy"],
        roberta_base["geometry"]["isotropy"],
        roberta_large["geometry"]["isotropy"]
    ]

    bars2 = ax2.bar(models, isotropies, color=colors)
    ax2.set_ylabel('Isotropy (λ_min/λ_max)')
    ax2.set_title('Representational Isotropy\n(Higher = More Uniform)')
    ax2.set_ylim(0, 0.04)

    # Add percentage change annotations
    ax2.annotate('+94%', xy=(1, isotropies[1]), xytext=(1, isotropies[1]+0.003),
                ha='center', fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_angular_separation.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_angular_separation.pdf', bbox_inches='tight')
    print(f"Saved: fig3_angular_separation.png/pdf")
    plt.close()


def figure4_summary(encoder_data, decoder_data):
    """Figure 4: Summary comparison of all findings."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Encoder data
    enc_results = sorted(encoder_data["results"], key=lambda x: x["n_params_millions"])
    enc_params = [r["n_params_millions"] for r in enc_results]
    enc_fishers = [r["fisher_criterion"] for r in enc_results]

    # Decoder data
    dec_results = sorted(decoder_data["results"], key=lambda x: x["n_params_millions"])
    dec_params = [r["n_params_millions"] for r in dec_results]
    dec_fishers = [r["fisher_criterion"] for r in dec_results]

    # Plot both
    ax.plot(enc_params, enc_fishers, 'o-', color='#2E86AB', linewidth=2,
            markersize=8, label='Encoders (BERT family)')
    ax.plot(dec_params, dec_fishers, 's-', color='#A23B72', linewidth=2,
            markersize=8, label='Decoders (GPT-2 family)')

    # Mark optimal points
    enc_best_idx = np.argmax(enc_fishers)
    dec_best_idx = np.argmax(dec_fishers)

    ax.scatter([enc_params[enc_best_idx]], [enc_fishers[enc_best_idx]],
               s=200, c='#2E86AB', edgecolors='gold', linewidths=3, zorder=5)
    ax.scatter([dec_params[dec_best_idx]], [dec_fishers[dec_best_idx]],
               s=200, c='#A23B72', edgecolors='gold', linewidths=3, zorder=5)

    # Annotate optimal points
    ax.annotate(f'Encoder optimal\n~{enc_params[enc_best_idx]:.0f}M',
                xy=(enc_params[enc_best_idx], enc_fishers[enc_best_idx]),
                xytext=(enc_params[enc_best_idx]*2, enc_fishers[enc_best_idx]+0.05),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='#2E86AB'))

    ax.annotate(f'Decoder optimal\n~{dec_params[dec_best_idx]:.0f}M',
                xy=(dec_params[dec_best_idx], dec_fishers[dec_best_idx]),
                xytext=(dec_params[dec_best_idx]*1.5, dec_fishers[dec_best_idx]+0.05),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='#A23B72'))

    ax.set_xlabel('Parameters (Millions)', fontsize=12)
    ax.set_ylabel("Fisher's Criterion (Phenomenal Discrimination)", fontsize=12)
    ax.set_title('The Optimal Size Phenomenon:\nArchitecture-Dependent Peak in Phenomenal Discrimination',
                fontsize=14)
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0.65, 1.25)

    # Add key insight text box
    textstr = 'Key Finding:\nOptimal size differs by architecture\n• Encoders: ~110M\n• Decoders: ~355M (~3x)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_summary.pdf', bbox_inches='tight')
    print(f"Saved: fig4_summary.png/pdf")
    plt.close()


def main():
    print("\n" + "="*60)
    print("   CREATING SCALING FIGURES")
    print("="*60 + "\n")

    # Load data
    print("Loading experimental data...")
    encoder_data, decoder_data, mechanistic_data = load_data()

    # Create figures
    print("\nGenerating figures...\n")

    figure1_encoder_scaling_curve(encoder_data)
    figure2_encoder_vs_decoder(encoder_data, decoder_data)
    figure3_angular_separation(mechanistic_data)
    figure4_summary(encoder_data, decoder_data)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
