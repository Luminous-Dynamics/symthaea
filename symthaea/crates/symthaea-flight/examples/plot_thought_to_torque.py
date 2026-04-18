#!/usr/bin/env python3
"""
Render a publication-quality plot of the HDC-LTC → motor command pipeline,
from the CSV produced by the `thought_to_torque` example.

Usage:

    # Step 1: produce the CSV
    cargo run -p symthaea-flight --example thought_to_torque --release \\
        -- TT_STEPS=400 TT_CSV=/tmp/thought_to_torque.csv

    # (or inline the env vars, same thing)
    TT_STEPS=400 TT_CSV=/tmp/thought_to_torque.csv \\
        cargo run -p symthaea-flight --example thought_to_torque --release

    # Step 2: render
    nix-shell -p 'python3.withPackages (ps: [ps.matplotlib ps.numpy])' \\
        --run "python3 plot_thought_to_torque.py /tmp/thought_to_torque.csv"

Produces `thought_to_torque.png` next to the input CSV.

CSV columns (from the example):
    step,phi,safety,motor_gain,hv_norm,hv_meanabs,
    raw_thrust,raw_roll,raw_pitch,raw_yaw,
    thrust,roll,pitch,yaw,
    scaled_thrust,scaled_roll,scaled_pitch,scaled_yaw
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TIER_COLORS = {
    "Green": "#2ecc71",
    "Yellow": "#f1c40f",
    "Orange": "#e67e22",
    "Red": "#e74c3c",
}


def load_csv(path: Path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def render(csv_path: Path, out_path: Path) -> None:
    rows = load_csv(csv_path)
    if not rows:
        print(f"no rows in {csv_path}", file=sys.stderr)
        sys.exit(1)

    steps = np.array([int(r["step"]) for r in rows])
    phi = np.array([float(r["phi"]) for r in rows])
    gain = np.array([float(r["motor_gain"]) for r in rows])
    tier = [r["safety"] for r in rows]
    thrust = np.array([float(r["thrust"]) for r in rows])
    scaled_thrust = np.array([float(r["scaled_thrust"]) for r in rows])
    hv_norm = np.array([float(r["hv_norm"]) for r in rows])

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2, 1]},
    )
    ax_phi, ax_thr, ax_hv, ax_tier = axes

    # ── Panel 1: phi trajectory with tier thresholds ──────────────────
    ax_phi.plot(steps, phi, color="#1f6ead", linewidth=1.1)
    for threshold, color in [
        (0.6, TIER_COLORS["Green"]),
        (0.3, TIER_COLORS["Yellow"]),
        (0.1, TIER_COLORS["Orange"]),
    ]:
        ax_phi.axhline(threshold, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
    ax_phi.set_ylabel(r"$\Phi$")
    ax_phi.set_ylim(0, 1)
    ax_phi.set_title(
        r"From thought to torque — $\Phi$-gated HDC-LTC → 4-DOF quadrotor command pipeline",
        fontsize=11,
    )
    ax_phi.grid(alpha=0.25)
    ax_phi.legend(
        ["$\\Phi$", "Green > 0.6", "Yellow > 0.3", "Orange > 0.1"],
        loc="upper left",
        fontsize=7,
        framealpha=0.85,
    )

    # ── Panel 2: raw vs scaled thrust ─────────────────────────────────
    ax_thr.plot(steps, thrust, color="#34495e", linewidth=0.9,
                label="controller output (raw)")
    ax_thr.plot(steps, scaled_thrust, color="#c0392b", linewidth=1.2,
                label=r"× $\Phi$ gain (envelope-scaled)")
    ax_thr.set_ylabel("thrust (N)")
    ax_thr.grid(alpha=0.25)
    ax_thr.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ── Panel 3: hv_norm (should stay near constant — controller normalizes)
    ax_hv.plot(steps, hv_norm, color="#8e44ad", linewidth=0.8, alpha=0.8)
    ax_hv.set_ylabel(r"$\|$hv$\|$ (16,384D)")
    ax_hv.grid(alpha=0.25)
    med = float(np.median(hv_norm))
    ax_hv.axhline(med, color="#8e44ad", linestyle=":", linewidth=0.8,
                   alpha=0.5, label=f"median = {med:.2f}")
    ax_hv.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ── Panel 4: tier colorband ───────────────────────────────────────
    # Paint a horizontal strip colored by the safety tier at each step.
    prev_tier = None
    start_idx = 0
    for i, t in enumerate(tier):
        if t != prev_tier and prev_tier is not None:
            ax_tier.axvspan(
                steps[start_idx],
                steps[i],
                color=TIER_COLORS.get(prev_tier, "#999"),
                alpha=0.6,
                linewidth=0,
            )
            start_idx = i
        prev_tier = t
    if prev_tier is not None:
        ax_tier.axvspan(
            steps[start_idx],
            steps[-1] + 1,
            color=TIER_COLORS.get(prev_tier, "#999"),
            alpha=0.6,
            linewidth=0,
        )
    ax_tier.set_xlim(steps[0], steps[-1])
    ax_tier.set_yticks([])
    ax_tier.set_ylabel("tier")
    ax_tier.set_xlabel("step")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(rows)} steps)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1
    out_path = csv_path.parent / (csv_path.stem + ".png")
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    render(csv_path, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
