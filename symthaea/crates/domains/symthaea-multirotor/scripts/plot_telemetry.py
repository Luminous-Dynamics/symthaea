#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Generate telemetry plots for the Symthaea flight paper.

Reads CSV telemetry from survival_reflex and kinetic_sacrifice examples
and produces publication-ready matplotlib figures.

Usage:
    python3 scripts/plot_telemetry.py [--survival FILE] [--sacrifice FILE] [--outdir DIR]
"""
import argparse
import csv
import os
import sys


def read_csv(path):
    """Read a CSV file into a list of dicts with float values."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_survival_reflex(data, outdir):
    """Plot survival reflex: position error + tau factor + free energy."""
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in data]
    pos_err = [r["position_error"] for r in data]
    fe = [r["free_energy"] for r in data]
    tau = [r["tau_factor"] for r in data]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Survival Reflex: +30% Mass Perturbation at Step 250", fontsize=14)

    # Position error
    ax1.plot(steps, pos_err, color="#2196F3", linewidth=1.0)
    ax1.axvline(x=250, color="#F44336", linestyle="--", alpha=0.7, label="Perturbation")
    ax1.set_ylabel("Position Error (m)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Free energy
    ax2.plot(steps, fe, color="#FF9800", linewidth=1.0)
    ax2.axvline(x=250, color="#F44336", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Free Energy")
    ax2.grid(True, alpha=0.3)

    # Tau factor
    ax3.plot(steps, tau, color="#4CAF50", linewidth=1.0)
    ax3.axvline(x=250, color="#F44336", linestyle="--", alpha=0.7)
    ax3.set_ylabel("Tau Factor")
    ax3.set_xlabel("Simulation Step")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "survival_reflex.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Also save SVG for paper
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle("Survival Reflex: +30% Mass Perturbation at Step 250", fontsize=14)
    ax1.plot(steps, pos_err, color="#2196F3", linewidth=1.0)
    ax1.axvline(x=250, color="#F44336", linestyle="--", alpha=0.7, label="Perturbation")
    ax1.set_ylabel("Position Error (m)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax2.plot(steps, fe, color="#FF9800", linewidth=1.0)
    ax2.axvline(x=250, color="#F44336", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Free Energy")
    ax2.grid(True, alpha=0.3)
    ax3.plot(steps, tau, color="#4CAF50", linewidth=1.0)
    ax3.axvline(x=250, color="#F44336", linestyle="--", alpha=0.7)
    ax3.set_ylabel("Tau Factor")
    ax3.set_xlabel("Simulation Step")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    svg_path = os.path.join(outdir, "survival_reflex.svg")
    fig2.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {svg_path}")


def plot_kinetic_sacrifice(data, outdir):
    """Plot kinetic sacrifice: drone trajectory + beam fall + danger signal."""
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in data]
    drone_z = [r["drone_z"] for r in data]
    beam_z = [r["beam_z"] for r in data]
    danger = [r["human_danger"] for r in data]
    fe = [r["free_energy"] for r in data]
    tau = [r["tau_factor"] for r in data]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Kinetic Sacrifice: FEP-Emergent Moral Reasoning", fontsize=14)

    # Altitude: drone vs beam
    axes[0].plot(steps, drone_z, color="#2196F3", linewidth=1.0, label="Drone altitude")
    axes[0].plot(steps, beam_z, color="#F44336", linewidth=1.0, label="Beam altitude")
    axes[0].axvline(x=400, color="#9C27B0", linestyle="--", alpha=0.7, label="Beam release")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Human danger + free energy
    ax2 = axes[1]
    color_fe = "#FF9800"
    color_danger = "#E91E63"
    ax2.plot(steps, fe, color=color_fe, linewidth=1.0, label="Free energy")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, danger, color=color_danger, linewidth=1.0, label="Human danger")
    ax2.set_ylabel("Free Energy", color=color_fe)
    ax2_twin.set_ylabel("Human Danger", color=color_danger)
    ax2.axvline(x=400, color="#9C27B0", linestyle="--", alpha=0.7)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Tau factor
    axes[2].plot(steps, tau, color="#4CAF50", linewidth=1.0)
    axes[2].axvline(x=400, color="#9C27B0", linestyle="--", alpha=0.7)
    axes[2].set_ylabel("Tau Factor")
    axes[2].set_xlabel("Simulation Step")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "kinetic_sacrifice.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # SVG
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle("Kinetic Sacrifice: FEP-Emergent Moral Reasoning", fontsize=14)
    axes2[0].plot(steps, drone_z, color="#2196F3", linewidth=1.0, label="Drone altitude")
    axes2[0].plot(steps, beam_z, color="#F44336", linewidth=1.0, label="Beam altitude")
    axes2[0].axvline(x=400, color="#9C27B0", linestyle="--", alpha=0.7, label="Beam release")
    axes2[0].set_ylabel("Altitude (m)")
    axes2[0].legend(loc="upper right", fontsize=9)
    axes2[0].grid(True, alpha=0.3)
    ax2b = axes2[1]
    ax2b.plot(steps, fe, color=color_fe, linewidth=1.0, label="Free energy")
    ax2b_twin = ax2b.twinx()
    ax2b_twin.plot(steps, danger, color=color_danger, linewidth=1.0, label="Human danger")
    ax2b.set_ylabel("Free Energy", color=color_fe)
    ax2b_twin.set_ylabel("Human Danger", color=color_danger)
    ax2b.axvline(x=400, color="#9C27B0", linestyle="--", alpha=0.7)
    lines1b, labels1b = ax2b.get_legend_handles_labels()
    lines2b, labels2b = ax2b_twin.get_legend_handles_labels()
    ax2b.legend(lines1b + lines2b, labels1b + labels2b, loc="upper right", fontsize=9)
    ax2b.grid(True, alpha=0.3)
    axes2[2].plot(steps, tau, color="#4CAF50", linewidth=1.0)
    axes2[2].axvline(x=400, color="#9C27B0", linestyle="--", alpha=0.7)
    axes2[2].set_ylabel("Tau Factor")
    axes2[2].set_xlabel("Simulation Step")
    axes2[2].grid(True, alpha=0.3)
    plt.tight_layout()
    svg_path = os.path.join(outdir, "kinetic_sacrifice.svg")
    fig2.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {svg_path}")


def plot_efe_ablation(data, outdir):
    """Plot EFE ablation study: precision sweep with decision boundary.

    Expects CSV with columns: safety_precision, efe_mission, efe_intercept, chose_intercept
    """
    import matplotlib.pyplot as plt
    import numpy as np

    precisions = [r["safety_precision"] for r in data]
    efe_mission = [r["efe_mission"] for r in data]
    efe_intercept = [r["efe_intercept"] for r in data]
    chose_intercept = [r["chose_intercept"] for r in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("EFE Ablation: Safety Precision Decision Boundary", fontsize=14)

    ax.plot(precisions, efe_mission, "o-", color="#F44336", linewidth=2, label="EFE(mission)")
    ax.plot(
        precisions,
        efe_intercept,
        "s-",
        color="#2196F3",
        linewidth=2,
        label="EFE(intercept)",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Safety Prior Precision (π_safety)", fontsize=12)
    ax.set_ylabel("Expected Free Energy", fontsize=12)

    # Shade decision regions
    mission_region = [p for p, c in zip(precisions, chose_intercept) if c < 0.5]
    intercept_region = [p for p, c in zip(precisions, chose_intercept) if c >= 0.5]
    if mission_region:
        ax.axvspan(
            min(precisions), max(mission_region) * 1.5,
            alpha=0.08, color="#F44336", label="MISSION region",
        )
    if intercept_region:
        ax.axvspan(
            min(intercept_region) * 0.7, max(precisions),
            alpha=0.08, color="#2196F3", label="INTERCEPT region",
        )

    # Find and mark crossover
    for i in range(1, len(data)):
        if chose_intercept[i] != chose_intercept[i - 1]:
            crossover_x = (precisions[i - 1] + precisions[i]) / 2
            crossover_y = (efe_mission[i] + efe_intercept[i]) / 2
            ax.axvline(
                x=crossover_x, color="#9C27B0", linestyle="--", linewidth=2,
                label=f"Crossover ≈ {crossover_x:.1f}",
            )
            break

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for fmt in ["png", "svg"]:
        path = os.path.join(outdir, f"efe_ablation.{fmt}")
        fig.savefig(path, dpi=150, bbox_inches="tight", format=fmt)
        print(f"  Saved: {path}")
    plt.close()


def plot_multi_scenario(data, outdir):
    """Plot multi-scenario comparison: grouped bar chart.

    Expects CSV with columns: variant, efe_mission, efe_intercept, chose_override, expected, matches
    """
    import matplotlib.pyplot as plt
    import numpy as np

    variants = [r["variant"] for r in data]
    efe_mission = [r["efe_mission"] for r in data]
    efe_intercept = [r["efe_intercept"] for r in data]
    matches = [r["matches"] for r in data]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Multi-Scenario EFE Comparison", fontsize=14)

    bars1 = ax.bar(x - width / 2, efe_mission, width, label="EFE(mission)", color="#F44336", alpha=0.8)
    bars2 = ax.bar(x + width / 2, efe_intercept, width, label="EFE(intercept)", color="#2196F3", alpha=0.8)

    # Mark correct/incorrect matches
    for i, m in enumerate(matches):
        marker = "✓" if m >= 0.5 else "✗"
        color = "#4CAF50" if m >= 0.5 else "#F44336"
        ax.text(i, max(efe_mission[i], efe_intercept[i]) * 1.05, marker,
                ha="center", va="bottom", fontsize=14, color=color, fontweight="bold")

    ax.set_xlabel("Scenario Variant", fontsize=12)
    ax.set_ylabel("Expected Free Energy", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for fmt in ["png", "svg"]:
        path = os.path.join(outdir, f"multi_scenario.{fmt}")
        fig.savefig(path, dpi=150, bbox_inches="tight", format=fmt)
        print(f"  Saved: {path}")
    plt.close()


def read_csv_strings(path):
    """Read a CSV file into a list of dicts, preserving strings and converting numbers."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot flight telemetry")
    parser.add_argument(
        "--survival",
        default="survival_reflex_telemetry.csv",
        help="Survival reflex CSV path",
    )
    parser.add_argument(
        "--sacrifice",
        default="kinetic_sacrifice_telemetry.csv",
        help="Kinetic sacrifice CSV path",
    )
    parser.add_argument(
        "--ablation",
        default=None,
        help="EFE ablation CSV path",
    )
    parser.add_argument(
        "--multi-scenario",
        default=None,
        help="Multi-scenario CSV path",
    )
    parser.add_argument(
        "--outdir",
        default="plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if os.path.exists(args.survival):
        print(f"Plotting survival reflex from {args.survival}...")
        data = read_csv(args.survival)
        plot_survival_reflex(data, args.outdir)
    else:
        print(f"Skipping survival reflex: {args.survival} not found")

    if os.path.exists(args.sacrifice):
        print(f"Plotting kinetic sacrifice from {args.sacrifice}...")
        data = read_csv(args.sacrifice)
        plot_kinetic_sacrifice(data, args.outdir)
    else:
        print(f"Skipping kinetic sacrifice: {args.sacrifice} not found")

    if args.ablation and os.path.exists(args.ablation):
        print(f"Plotting EFE ablation from {args.ablation}...")
        data = read_csv(args.ablation)
        plot_efe_ablation(data, args.outdir)
    elif args.ablation:
        print(f"Skipping EFE ablation: {args.ablation} not found")

    multi_scenario = getattr(args, "multi_scenario", None)
    if multi_scenario and os.path.exists(multi_scenario):
        print(f"Plotting multi-scenario from {multi_scenario}...")
        data = read_csv_strings(multi_scenario)
        plot_multi_scenario(data, args.outdir)
    elif multi_scenario:
        print(f"Skipping multi-scenario: {multi_scenario} not found")

    print("\nDone!")


if __name__ == "__main__":
    main()
