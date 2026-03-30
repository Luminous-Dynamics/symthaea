#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Augment 20-channel Broca training data with 4 new channels (time_pressure,
domain_familiarity, social_context, response_confidence) to produce 24-channel data.

New channel values are derived from existing channels to create realistic correlations:
- time_pressure: derived from arousal (ch2) + noise
- domain_familiarity: derived from epistemic (ch0) inverted + noise
- social_context: derived from warmth (ch3) + noise
- response_confidence: derived from epistemic (ch0) inverted + coherence-like signal

Usage:
    python3 augment-channels-24.py input.jsonl output.jsonl [--seed 42]
"""

import json
import sys
import random

def augment_channels(channels_20, rng):
    """Add 4 new channels derived from existing ones with realistic correlations."""
    assert len(channels_20) == 20, f"Expected 20 channels, got {len(channels_20)}"

    epistemic = channels_20[0]   # 0=Certain, 4=OutOfDomain
    valence = channels_20[1]     # -1..1
    arousal = channels_20[2]     # 0..1
    warmth = channels_20[3]      # 0..1
    psi = channels_20[4]         # consciousness level 0..1

    # time_pressure: correlated with arousal (high arousal = more time pressure)
    # Range: 0..1 where 0=relaxed, 1=urgent
    time_pressure = max(0.0, min(1.0, arousal * 0.6 + rng.gauss(0, 0.15)))

    # domain_familiarity: inverse of epistemic uncertainty
    # epistemic 0=Certain→familiarity~1, epistemic 4=OutOfDomain→familiarity~0
    domain_familiarity = max(0.0, min(1.0, 1.0 - epistemic / 4.0 + rng.gauss(0, 0.1)))

    # social_context: correlated with warmth (warm = social)
    # Range: 0..1 where 0=solo, 1=social
    social_context = max(0.0, min(1.0, warmth * 0.7 + 0.15 + rng.gauss(0, 0.12)))

    # response_confidence: derived from epistemic certainty and consciousness
    # Higher confidence when certain and conscious
    base_confidence = (1.0 - epistemic / 4.0) * 0.6 + psi * 0.3
    response_confidence = max(0.0, min(1.0, base_confidence + rng.gauss(0, 0.1)))

    return channels_20 + [time_pressure, domain_familiarity, social_context, response_confidence]


def main():
    if len(sys.argv) < 3:
        print("Usage: augment-channels-24.py input.jsonl output.jsonl [--seed 42]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    seed = 42
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed = int(sys.argv[idx + 1])

    rng = random.Random(seed)

    count_20 = 0
    count_24 = 0
    count_other = 0

    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            channels = record["channels"]

            if len(channels) == 20:
                record["channels"] = augment_channels(channels, rng)
                count_20 += 1
            elif len(channels) == 24:
                count_24 += 1  # Already 24 channels, pass through
            else:
                count_other += 1
                # Pad or truncate to 24
                if len(channels) < 24:
                    record["channels"] = channels + [0.0] * (24 - len(channels))
                else:
                    record["channels"] = channels[:24]

            f_out.write(json.dumps(record) + "\n")

    print(f"Augmented: {count_20} (20→24), passed: {count_24} (already 24), other: {count_other}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
