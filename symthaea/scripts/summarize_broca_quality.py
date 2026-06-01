#!/usr/bin/env python3
"""Print a compact human-readable summary for a Broca quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric(report: dict[str, Any], section: str, name: str) -> Any:
    value = report.get(section, {})
    if isinstance(value, dict):
        return value.get(name)
    return None


def collapse_line(report: dict[str, Any], section: str) -> str:
    collapse = metric(report, section, "top_token_collapse")
    rate = metric(report, section, "top_token_collapse_rate")
    if not isinstance(collapse, dict):
        return f"{section}: collapse_rate={fmt(rate)}"
    token = collapse.get("token")
    token_id = collapse.get("token_id")
    count = collapse.get("count")
    total = collapse.get("total")
    return (
        f"{section}: collapse_rate={fmt(rate)} "
        f"token={token!r} token_id={fmt(token_id)} count={fmt(count)}/{fmt(total)}"
    )


def summarize(report: dict[str, Any]) -> list[str]:
    metadata = report.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    failures = report.get("quality_gate_failures") or []
    status = report.get("threshold_status")
    if status is None:
        status = "report-only" if not failures else "failed"

    lines = [
        f"[broca-quality] status={status} backend={metadata.get('backend', 'n/a')} lane={metadata.get('eval_lane', 'n/a')}",
        (
            "[broca-quality] gated: "
            f"ppl={fmt(metric(report, 'gated_generation', 'perplexity'))} "
            f"coh={fmt(metric(report, 'gated_generation', 'avg_coherence'))} "
            f"english={fmt(metric(report, 'gated_generation', 'english_word_ratio'))}"
        ),
        (
            "[broca-quality] delta: "
            f"ppl={fmt(metric(report, 'delta', 'perplexity'))} "
            f"coh={fmt(metric(report, 'delta', 'avg_coherence'))} "
            f"collapse={fmt(metric(report, 'delta', 'top_token_collapse_rate'))}"
        ),
        f"[broca-quality] {collapse_line(report, 'raw_generation')}",
        f"[broca-quality] {collapse_line(report, 'gated_generation')}",
    ]

    if failures:
        for failure in failures:
            if isinstance(failure, dict):
                lines.append(
                    "[broca-quality] failure: "
                    f"{failure.get('metric', 'unknown')} observed={fmt(failure.get('observed'))} "
                    f"threshold={fmt(failure.get('threshold'))}"
                )
            else:
                lines.append(f"[broca-quality] failure: {failure}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    for line in summarize(report):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
