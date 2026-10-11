#!/usr/bin/env python3
"""SPINE-000C: static, read-only census of explicit manager/inline overlap witnesses.

This tool does not execute cognition and does not infer causal load. It only records
source-level statements that explicitly describe dual-write / inline coexistence so
later dynamic instrumentation can target the correct seams without guessing from names.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from spine_census import git_value, iter_manager_records


WITNESS_PATTERNS = (
    "dual-write",
    "does not replace inline",
    "alongside existing inline",
    "stays inline",
    "remains inline",
    "inline in cycle",
    "direct mutation",
    "direct mutations",
)
BACKTICK_RE = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class OverlapRecord:
    subsystem_type: str
    subsystem_name: str | None
    source_path: str
    declares_dual_write: bool
    overlap_status: str
    source_witnesses: list[str]
    explicit_inline_refs: list[str]


def _clean_witness(line: str) -> str:
    text = line.strip()
    while text.startswith("/"):
        text = text[1:].lstrip()
    if text.startswith("!"):
        text = text[1:].lstrip()
    return text


def _looks_like_inline_ref(token: str) -> bool:
    lowered = token.lower()
    return (
        ".rs" in lowered
        or "cycle_phase" in lowered
        or "cognitiveloopservice" in lowered
        or "brocamanager" in lowered
        or "inline" in lowered
    )


def build_report(root: Path) -> dict[str, object]:
    manager_records = list(iter_manager_records(root))
    overlaps: list[OverlapRecord] = []

    for record in manager_records:
        path = root / record.source_path
        text = path.read_text(encoding="utf-8")

        witnesses: list[str] = []
        refs: set[str] = set()
        for line in text.splitlines():
            lowered = line.lower()
            if any(pattern in lowered for pattern in WITNESS_PATTERNS):
                cleaned = _clean_witness(line)
                if cleaned:
                    witnesses.append(cleaned)
                for token in BACKTICK_RE.findall(line):
                    if _looks_like_inline_ref(token):
                        refs.add(token.strip())

        if record.declares_dual_write:
            status = "EXPLICIT_DUAL_WRITE_WITNESS"
        elif witnesses:
            status = "EXPLICIT_INLINE_COEXISTENCE_WITNESS"
        else:
            status = "NO_EXPLICIT_SOURCE_WITNESS"

        overlaps.append(
            OverlapRecord(
                subsystem_type=record.subsystem_type,
                subsystem_name=record.subsystem_name,
                source_path=record.source_path,
                declares_dual_write=record.declares_dual_write,
                overlap_status=status,
                source_witnesses=witnesses,
                explicit_inline_refs=sorted(refs),
            )
        )

    explicit_count = sum(
        1 for record in overlaps if record.overlap_status != "NO_EXPLICIT_SOURCE_WITNESS"
    )

    return {
        "schema": "symthaea.spine.census.explicit-overlap.v1",
        "scope": "explicit source-level manager/inline coexistence witnesses only",
        "authority": "measurement-only",
        "causal_load_claimed": False,
        "absence_of_witness_means_no_overlap": False,
        "git_head": git_value(root, "rev-parse", "HEAD"),
        "git_index_tree": git_value(root, "write-tree"),
        "manager_count": len(overlaps),
        "explicit_overlap_count": explicit_count,
        "records": [asdict(record) for record in overlaps],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Symthaea repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Without this flag, writes only to stdout.",
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    report = build_report(root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
