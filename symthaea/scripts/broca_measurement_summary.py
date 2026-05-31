#!/usr/bin/env python3
"""Generate a human-readable Broca measurement summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            pairs[key] = value
    return pairs


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: broca_measurement_summary.py ARTIFACT_DIR", file=sys.stderr)
        return 2

    artifact_dir = Path(sys.argv[1])
    decoder = load_json(artifact_dir / "decoder-ab.json") or {}
    exercism = load_json(artifact_dir / "exercism-bench.json") or {}
    compare = load_json(artifact_dir / "checkpoint-compare.json")
    run_manifest = manifest(artifact_dir / "measurement-manifest.env")

    decoder_agg = decoder.get("aggregate", {})
    decoder_gates = decoder.get("gates", {})
    compare_promotion = (compare or {}).get("promotion", {})

    lines = [
        "# Broca Measurement Summary",
        "",
        f"- Artifact directory: `{artifact_dir}`",
        f"- Git rev: `{run_manifest.get('git_rev', 'unknown')}`",
        f"- Created UTC: `{run_manifest.get('created_at_utc', 'unknown')}`",
        f"- Decoder evidence: `{decoder.get('evidence_level', 'missing')}`",
        f"- Exercism evidence: `{exercism.get('evidence_level', 'missing')}`",
        "",
        "## Decoder A/B",
        "",
        f"- Gates passed: {fmt(decoder_gates.get('passed'))}",
        f"- Avg direct semantic drift: {fmt(decoder_agg.get('avg_direct_semantic_drift'))}",
        f"- Direct hallucination rate: {fmt(decoder_agg.get('direct_hallucination_rate'))}",
        f"- Avg structured confidence: {fmt(decoder_agg.get('avg_structured_confidence'))}",
        f"- Avg structured intensity: {fmt(decoder_agg.get('avg_structured_intensity'))}",
        f"- Avg structured validity: {fmt(decoder_agg.get('avg_structured_validity'))}",
        f"- Structured required-role rate: {fmt(decoder_agg.get('structured_required_role_rate'))}",
        f"- Avg Mamba semantic drift: {fmt(decoder_agg.get('avg_mamba_semantic_drift'))}",
        f"- Mamba hallucination rate: {fmt(decoder_agg.get('mamba_hallucination_rate'))}",
        "",
        "## Exercism",
        "",
        f"- Measured: {fmt(exercism.get('measured'))}",
        f"- Total exercises: {fmt(exercism.get('total_exercises'))}",
        f"- Compile successes: {fmt(exercism.get('compile_successes'))}",
        f"- Test successes: {fmt(exercism.get('test_successes'))}",
    ]

    failures = decoder_gates.get("failures") or []
    if failures:
        lines += ["", "### Decoder Gate Failures", ""]
        for failure in failures:
            lines.append(
                f"- `{failure.get('metric')}` observed {fmt(failure.get('observed'))}, "
                f"threshold {fmt(failure.get('threshold'))}"
            )

    if compare is not None:
        lines += [
            "",
            "## Promotion Comparison",
            "",
            f"- Promotion passed: {fmt(compare_promotion.get('passed'))}",
        ]
        for failure in compare_promotion.get("failures") or []:
            lines.append(
                f"- `{failure.get('metric')}` regressed by {fmt(failure.get('delta'))} "
                f"(allowed {fmt(failure.get('allowed_regression'))})"
            )
        missing_required = compare_promotion.get("missing_required_metrics") or []
        if missing_required:
            lines.append(
                "- Missing required metrics: "
                + ", ".join(f"`{metric}`" for metric in missing_required)
            )
        missing = compare.get("missing_metrics") or []
        if missing:
            lines.append(
                "- Missing optional metrics: "
                + ", ".join(f"`{metric}`" for metric in missing)
            )

    lines += [
        "",
        "## Measurement Knobs",
        "",
    ]
    for key in sorted(run_manifest):
        if key not in {"schema_version", "created_at_utc", "git_rev"}:
            lines.append(f"- `{key}`: `{run_manifest[key]}`")

    (artifact_dir / "measurement-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
