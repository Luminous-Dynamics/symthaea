#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Aggregate BFT matrix JSON artifacts into a single summary file.

Usage:
    nix develop --command poetry run python scripts/generate_bft_matrix.py

Produces:
    0TML/tests/results/bft_matrix.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    results_dir = Path("0TML/tests/results")
    artefacts = sorted(results_dir.glob("bft_results_*_byz.json"))

    if not artefacts:
        raise SystemExit("No bft_results_*_byz.json files found – run the matrix tests first.")

    matrix_rows = []
    for artefact in artefacts:
        with artefact.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        try:
            pct = int(artefact.stem.split("_")[2])
        except (IndexError, ValueError) as err:
            raise ValueError(f"Unexpected filename format: {artefact.name}") from err

        matrix_rows.append({
            "byzantine_percent": pct,
            "detection_rate": payload["detection_rate"],
            "false_positive_rate": payload["false_positive_rate"],
            "accuracy": payload["accuracy"],
            "precision": payload["precision"],
            "recall": payload["recall"],
            "true_positives": payload["true_positives"],
            "false_positives": payload["false_positives"],
            "true_negatives": payload["true_negatives"],
            "false_negatives": payload["false_negatives"],
            "source": artefact.name,
        })

    output = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "matrix": matrix_rows,
    }

    output_path = results_dir / "bft_matrix.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"✅ Wrote {output_path}")


if __name__ == "__main__":
    main()
