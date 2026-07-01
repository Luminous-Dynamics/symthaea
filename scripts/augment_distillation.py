#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Augment distillation dataset with noise perturbations.

Reads a distillation JSONL file and creates augmented copies with:
1. Channel noise perturbation (±noise_scale on each channel)
2. Emotional valence/arousal flips
3. Epistemic confidence variation
4. Consciousness level variation

The target text stays the same — the goal is to teach the projection
that similar thoughts → similar SSM representations (robustness).

Usage:
    python augment_distillation.py INPUT.jsonl OUTPUT.jsonl [--multiplier 3] [--noise 0.1]
"""

import json
import random
import sys
from pathlib import Path


def augment_channels(channels: list[float], noise_scale: float, rng: random.Random) -> list[float]:
    """Add Gaussian noise to each channel, clamping to valid ranges."""
    augmented = []
    for i, v in enumerate(channels):
        noise = rng.gauss(0, noise_scale)
        new_v = v + noise
        # Channel-specific clamping
        if i == 0:
            # Intent (one-hot-ish) — keep in [0, 1]
            new_v = max(0.0, min(1.0, new_v))
        elif i in (9, 10, 11):
            # Valence [-1, 1], arousal [0, 1], warmth [0, 1]
            if i == 9:
                new_v = max(-1.0, min(1.0, new_v))
            else:
                new_v = max(0.0, min(1.0, new_v))
        elif i in (12, 13, 14):
            # Consciousness (psi, meta, coherence) [0, 1]
            new_v = max(0.0, min(1.0, new_v))
        else:
            # General clamp
            new_v = max(-1.0, min(1.0, new_v))
        augmented.append(round(new_v, 6))
    return augmented


def flip_emotion(channels: list[float]) -> list[float]:
    """Flip emotional valence while keeping everything else."""
    ch = channels.copy()
    if len(ch) > 9:
        ch[9] = -ch[9]  # Negate valence
    return ch


def vary_consciousness(channels: list[float], rng: random.Random) -> list[float]:
    """Randomly vary consciousness level."""
    ch = channels.copy()
    if len(ch) > 12:
        # Psi
        ch[12] = max(0.0, min(1.0, rng.uniform(0.3, 1.0)))
        # Meta-awareness
        ch[13] = max(0.0, min(1.0, rng.uniform(0.2, 0.9)))
        # Coherence
        ch[14] = max(0.0, min(1.0, rng.uniform(0.3, 0.95)))
    return ch


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    multiplier = 3
    noise_scale = 0.1

    for i, arg in enumerate(sys.argv[3:], 3):
        if arg == "--multiplier" and i + 1 < len(sys.argv):
            multiplier = int(sys.argv[i + 1])
        elif arg == "--noise" and i + 1 < len(sys.argv):
            noise_scale = float(sys.argv[i + 1])

    rng = random.Random(42)

    # Load original data
    originals = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                originals.append(json.loads(line))

    print(f"Loaded {len(originals)} original samples")

    # Generate augmented samples
    augmented = []
    for entry in originals:
        channels = entry["channels"]
        text = entry["target_text"]
        token_ids = entry.get("target_ids", [])

        for _ in range(multiplier):
            strategy = rng.choice(["noise", "noise", "consciousness", "flip_emotion"])

            if strategy == "noise":
                new_ch = augment_channels(channels, noise_scale, rng)
            elif strategy == "consciousness":
                new_ch = vary_consciousness(
                    augment_channels(channels, noise_scale * 0.5, rng), rng
                )
            elif strategy == "flip_emotion":
                new_ch = flip_emotion(
                    augment_channels(channels, noise_scale * 0.3, rng)
                )
            else:
                new_ch = augment_channels(channels, noise_scale, rng)

            augmented.append({
                "channels": new_ch,
                "target_text": text,
                "target_ids": token_ids,
            })

    print(f"Generated {len(augmented)} augmented samples (multiplier={multiplier})")

    # Write combined dataset (originals + augmented)
    total = 0
    with open(output_path, "w") as f:
        for entry in originals:
            f.write(json.dumps(entry) + "\n")
            total += 1
        for entry in augmented:
            f.write(json.dumps(entry) + "\n")
            total += 1

    print(f"Wrote {total} total samples to {output_path}")
    print(f"  Original: {len(originals)}")
    print(f"  Augmented: {len(augmented)}")


if __name__ == "__main__":
    main()
