#!/usr/bin/env python3
"""SPINE-000A: static, read-only census of CognitiveSubsystem managers.

This tool does not execute cognition and does not classify causal load.
It inventories the manager layer so later dynamic receipts can be bound
to a stable subsystem identity/schedule/proposal surface.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


IMPL_RE = re.compile(r"impl\s+CognitiveSubsystem\s+for\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
NAME_RE = re.compile(
    r"fn\s+name\s*\(\s*&self\s*\)\s*->\s*&'static\s+str\s*\{\s*\"([^\"]+)\"",
    re.MULTILINE,
)
INTERVAL_RE = re.compile(
    r"fn\s+interval\s*\(\s*&self\s*\)\s*->\s*u32\s*\{\s*([^}\n]+)",
    re.MULTILINE,
)

PROPOSAL_FIELDS = (
    "confidence_delta",
    "lr_modulation",
    "exploration_delta",
    "arousal_delta",
    "valence_delta",
)
FLAG_NAMES = (
    "REQUEST_EXPLORATION",
    "REQUEST_CONSOLIDATION",
    "ANOMALY_DETECTED",
    "VETO_ACTION",
    "REQUEST_REST",
    "HAS_TELEMETRY",
    "REQUEST_BROADCAST",
    "ESCALATE_URGENCY",
    "REQUEST_GEODESIC",
)


@dataclass(frozen=True)
class ManagerRecord:
    subsystem_type: str
    subsystem_name: str | None
    module: str
    source_path: str
    interval_expr: str | None
    feature_gate: str | None
    proposal_fields: list[str]
    output_flags: list[str]
    declares_dual_write: bool


def extract_braced_block(text: str, open_brace: int) -> str:
    depth = 0
    for i in range(open_brace, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : i + 1]
    raise ValueError("unterminated impl block")


def parse_module_feature_gates(mod_text: str) -> dict[str, str | None]:
    """Capture immediately preceding cfg attributes for `mod foo;` declarations."""
    gates: dict[str, str | None] = {}
    pending_attrs: list[str] = []
    for line in mod_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#[cfg("):
            pending_attrs.append(stripped)
            continue

        match = re.match(
            r"pub(?:\(crate\))?\s+mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
            stripped,
        )
        if match:
            module = match.group(1)
            gates[module] = " && ".join(pending_attrs) if pending_attrs else None
            pending_attrs.clear()
            continue

        if stripped and not stripped.startswith("//") and not stripped.startswith("#["):
            pending_attrs.clear()
    return gates


def git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def iter_manager_records(root: Path) -> Iterable[ManagerRecord]:
    managers_dir = root / "src" / "cognitive_loop" / "managers"
    mod_path = managers_dir / "mod.rs"
    if not managers_dir.is_dir():
        raise FileNotFoundError(f"manager directory not found: {managers_dir}")

    module_gates = (
        parse_module_feature_gates(mod_path.read_text(encoding="utf-8"))
        if mod_path.exists()
        else {}
    )

    for path in sorted(managers_dir.glob("*.rs")):
        if path.name == "mod.rs":
            continue
        text = path.read_text(encoding="utf-8")
        module = path.stem

        for match in IMPL_RE.finditer(text):
            open_brace = text.find("{", match.start())
            block = extract_braced_block(text, open_brace)
            name_match = NAME_RE.search(block)
            interval_match = INTERVAL_RE.search(block)

            fields = [field for field in PROPOSAL_FIELDS if re.search(rf"\b{field}\b", block)]
            flags = [flag for flag in FLAG_NAMES if re.search(rf"\b{flag}\b", block)]

            lowered = text.lower()
            dual_write = (
                "dual-write" in lowered
                or "does not replace inline" in lowered
                or "alongside existing inline" in lowered
            )

            yield ManagerRecord(
                subsystem_type=match.group(1),
                subsystem_name=name_match.group(1) if name_match else None,
                module=module,
                source_path=str(path.relative_to(root)),
                interval_expr=interval_match.group(1).strip() if interval_match else None,
                feature_gate=module_gates.get(module),
                proposal_fields=fields,
                output_flags=flags,
                declares_dual_write=dual_write,
            )


def build_report(root: Path) -> dict[str, object]:
    records = list(iter_manager_records(root))
    names = [r.subsystem_name for r in records if r.subsystem_name]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    return {
        "schema": "symthaea.spine.census.static-registry.v1",
        "scope": "src/cognitive_loop/managers/*.rs",
        "authority": "measurement-only",
        "causal_load_claimed": False,
        "git_head": git_value(root, "rev-parse", "HEAD"),
        "git_index_tree": git_value(root, "write-tree"),
        "manager_count": len(records),
        "duplicate_subsystem_names": duplicates,
        "managers": [asdict(record) for record in records],
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

    if report["duplicate_subsystem_names"]:
        raise SystemExit(
            "duplicate CognitiveSubsystem names: "
            + ", ".join(report["duplicate_subsystem_names"])
        )

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
