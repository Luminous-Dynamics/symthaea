#!/usr/bin/env python3
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
    axes[0].axvline(x=300, color="#9C27B0", linestyle="--", alpha=0.7, label="Beam release")
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
    ax2.axvline(x=300, color="#9C27B0", linestyle="--", alpha=0.7)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Tau factor
    axes[2].plot(steps, tau, color="#4CAF50", linewidth=1.0)
    axes[2].axvline(x=300, color="#9C27B0", linestyle="--", alpha=0.7)
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
    axes2[0].axvline(x=300, color="#9C27B0", linestyle="--", alpha=0.7, label="Beam release")
    axes2[0].set_ylabel("Altitude (m)")
    axes2[0].legend(loc="upper right", fontsize=9)
    axes2[0].grid(True, alpha=0.3)
    ax2b = axes2[1]
    ax2b.plot(steps, fe, color=color_fe, linewidth=1.0, label="Free energy")
    ax2b_twin = ax2b.twinx()
    ax2b_twin.plot(steps, danger, color=color_danger, linewidth=1.0, label="Human danger")
    ax2b.set_ylabel("Free Energy", color=color_fe)
    ax2b_twin.set_ylabel("Human Danger", color=color_danger)
    ax2b.axvline(x=300, color="#9C27B0", linestyle="--", alpha=0.7)
    lines1b, labels1b = ax2b.get_legend_handles_labels()
    lines2b, labels2b = ax2b_twin.get_legend_handles_labels()
    ax2b.legend(lines1b + lines2b, labels1b + labels2b, loc="upper right", fontsize=9)
    ax2b.grid(True, alpha=0.3)
    axes2[2].plot(steps, tau, color="#4CAF50", linewidth=1.0)
    axes2[2].axvline(x=300, color="#9C27B0", linestyle="--", alpha=0.7)
    axes2[2].set_ylabel("Tau Factor")
    axes2[2].set_xlabel("Simulation Step")
    axes2[2].grid(True, alpha=0.3)
    plt.tight_layout()
    svg_path = os.path.join(outdir, "kinetic_sacrifice.svg")
    fig2.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {svg_path}")


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

    print("\nDone!")


if __name__ == "__main__":
    main()
