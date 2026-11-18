#!/usr/bin/env python3
"""Export label-free D4 structural inputs for external proposal methods.

This creates adapter inputs only. The repository also contains public oracle
truth in cases.json, so these exports are NOT a sealed blind benchmark set.
Use them to prevent accidental target-field consumption by adapters; use a
separate protected evaluator for genuinely blind external evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "fixtures" / "v0.1" / "cases.json"
OUTPUT = ROOT / "public_inputs" / "v0.1"


def public_case(case: dict) -> dict:
    result = {
        "case_id": case["case_id"],
        "split": case["split"],
        "track": case["track"],
    }
    if "geometry" in case:
        result["geometry"] = case["geometry"]
    if "polynomial" in case:
        result["polynomial"] = case["polynomial"]
    return result


def export(manifest: dict) -> dict[str, dict]:
    exports = {}
    for split in ("discovery", "calibration", "final_holdout"):
        cases = [
            public_case(case)
            for case in manifest["cases"]
            if case.get("controls", {}).get("generated") is True
            and case["split"] == split
            and case["track"] == "exact_finite_structural_stabilizer"
        ]
        cases.sort(key=lambda case: case["case_id"])
        if len(cases) != 300:
            raise ValueError(f"{split}: expected 300 generated inputs, found {len(cases)}")
        for case in cases:
            forbidden = {"target_subgroup", "expected_subgroup_label", "controls"}
            if forbidden.intersection(case):
                raise ValueError(f"{case['case_id']}: target metadata leaked into public case")
            if not case["case_id"].startswith("CASE-"):
                raise ValueError(f"{case['case_id']}: generated case ID is not opaque")
        exports[split] = {
            "schema_version": "0.1.0",
            "benchmark_id": manifest["benchmark_id"],
            "benchmark_track": "exact_finite_structural_stabilizer",
            "split": split,
            "cases": cases,
        }
    return exports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exports = export(manifest)
    if args.write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for split, payload in exports.items():
            target = OUTPUT / f"{split}.json"
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    else:
        for split, payload in exports.items():
            print(f"{split}: {len(payload['cases'])} label-free cases")
    print("PUBLIC_INPUT_EXPORT_PASS: label-free fields; not a sealed blind set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
