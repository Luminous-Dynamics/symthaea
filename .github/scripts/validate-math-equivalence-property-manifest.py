#!/usr/bin/env python3
"""Validate the frozen MATH-REP-001D0 property-generator manifest.

Stdlib-only by design. This checks configuration identity/drift, not mathematical
correctness of generated cases and not the behavior of any normalizer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data/benchmarks/math_equivalence_property_q0_v2.json"
DEFAULT_SOURCE = (
    ROOT
    / "crates/core/symthaea-core/examples/support/math_equivalence_property_q0_v2.rs"
)

HEX_RE = re.compile(r"^[0-9a-f]{16}$")


def fail(message: str) -> None:
    raise ValueError(message)


def rust_u64_seeds(source: str, const_name: str) -> list[str]:
    match = re.search(
        rf"pub const {re.escape(const_name)}:\s*\[u64;\s*\d+\]\s*=\s*\[(.*?)\];",
        source,
        flags=re.S,
    )
    if not match:
        fail(f"missing Rust seed constant {const_name}")
    raw = re.findall(r"0x([0-9A-Fa-f_]+)", match.group(1))
    return [item.replace("_", "").lower() for item in raw]


def rust_usize_const(source: str, const_name: str) -> int:
    match = re.search(
        rf"pub const {re.escape(const_name)}:\s*usize\s*=\s*(\d+)\s*;",
        source,
    )
    if not match:
        fail(f"missing Rust usize constant {const_name}")
    return int(match.group(1))


def rust_string_const(source: str, const_name: str) -> str:
    match = re.search(
        rf'pub const {re.escape(const_name)}:\s*&str\s*=\s*"([^"]+)"\s*;',
        source,
    )
    if not match:
        fail(f"missing Rust string constant {const_name}")
    return match.group(1)


def rust_enum_cycle(source: str, enum_name: str, size: int) -> list[str]:
    match = re.search(
        rf"const FAMILIES:\s*\[{re.escape(enum_name)};\s*{size}\]\s*=\s*\[(.*?)\];",
        source,
        flags=re.S,
    )
    if not match:
        fail(f"missing {enum_name} FAMILIES[{size}] cycle")
    items = []
    for raw in match.group(1).split(","):
        item = raw.strip()
        if item:
            items.append(item.split("::")[-1])
    return items


def validate(manifest_path: Path, source_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = source_path.read_text(encoding="utf-8")

    generator_id = manifest.get("generator_id")
    if generator_id != "math-equivalence-property-q0-v2":
        fail(f"unexpected generator_id: {generator_id!r}")
    if manifest.get("version") != generator_id:
        fail("version must equal generator_id")
    if manifest.get("authority") != "MeasurementOnly":
        fail("authority must remain MeasurementOnly")

    if rust_string_const(source, "GENERATOR_ID") != generator_id:
        fail("Rust GENERATOR_ID disagrees with manifest")
    if rust_string_const(source, "AUTHORITY") != manifest["authority"]:
        fail("Rust AUTHORITY disagrees with manifest")

    dev = manifest.get("dev_seeds_hex")
    evaluation = manifest.get("evaluation_seeds_hex")
    if not isinstance(dev, list) or not isinstance(evaluation, list):
        fail("seed lists must be arrays")
    for label, seeds in (("dev", dev), ("evaluation", evaluation)):
        if len(seeds) != len(set(seeds)):
            fail(f"duplicate {label} seed")
        for seed in seeds:
            if not isinstance(seed, str) or not HEX_RE.fullmatch(seed):
                fail(f"invalid {label} seed: {seed!r}")
    if set(dev) & set(evaluation):
        fail("dev and evaluation seeds must be disjoint")
    if rust_u64_seeds(source, "DEV_SEEDS") != dev:
        fail("Rust DEV_SEEDS disagree with manifest")
    if rust_u64_seeds(source, "EVAL_SEEDS") != evaluation:
        fail("Rust EVAL_SEEDS disagree with manifest")

    pairs = manifest.get("pairs_per_seed")
    refusals = manifest.get("refusals_per_seed")
    if pairs != rust_usize_const(source, "PAIRS_PER_SEED"):
        fail("PAIRS_PER_SEED drift")
    if refusals != rust_usize_const(source, "REFUSALS_PER_SEED"):
        fail("REFUSALS_PER_SEED drift")

    expected_cases = len(dev) * (pairs + refusals)
    if manifest.get("cases_per_split") != expected_cases:
        fail(
            f"cases_per_split={manifest.get('cases_per_split')} but expected {expected_cases}"
        )
    if len(evaluation) != len(dev):
        fail("dev/evaluation seed counts must match for v2")

    pair_cycle = manifest.get("pair_family_cycle")
    if pair_cycle != rust_enum_cycle(source, "OracleFamily", 24):
        fail("pair family order drift between Rust and manifest")
    refusal_cycle = manifest.get("refusal_family_cycle")
    if refusal_cycle != rust_enum_cycle(source, "RefusalFamily", 8):
        fail("refusal family order drift between Rust and manifest")

    same = manifest.get("same_normal_form_families")
    different = manifest.get("different_normal_form_families")
    if len(same) != len(set(same)) or len(different) != len(set(different)):
        fail("family partitions must not contain duplicates")
    if set(same) & set(different):
        fail("same/different family partitions overlap")
    if set(same) | set(different) != set(pair_cycle):
        fail("same/different family partitions must cover pair cycle exactly")
    if len(same) != 18 or len(different) != 6:
        fail("v2 family partition must remain 18 same / 6 different")

    full_pair_cycles, remainder = divmod(pairs, len(pair_cycle))
    expected_same_per_seed = full_pair_cycles * len(same) + sum(
        1 for family in pair_cycle[:remainder] if family in set(same)
    )
    expected_different_per_seed = pairs - expected_same_per_seed
    if expected_same_per_seed != 98 or expected_different_per_seed != 30:
        fail(
            "frozen pair balance changed: expected 98 same / 30 different per seed"
        )

    if refusals % len(refusal_cycle) != 0:
        fail("refusal count must be an exact multiple of refusal family cycle")
    refusal_family_per_seed = refusals // len(refusal_cycle)
    if refusal_family_per_seed != 3:
        fail("each refusal family must appear exactly 3 times per seed")

    rules = manifest.get("oracle_rules", {})
    if rules.get("labels_from_normalizer_under_test") is not False:
        fail("normalizer under test must not generate labels")
    if rules.get("labels_from_retriever_under_test") is not False:
        fail("retriever under test must not generate labels")
    if rules.get("side_condition_sensitive_division_or_cancellation") is not False:
        fail("side-condition-sensitive division/cancellation must remain disabled")

    policy = manifest.get("evaluation_policy", {})
    if policy.get("evaluation_seeds_must_not_be_used_to_tune_v2") is not True:
        fail("evaluation seeds must remain non-tuning")
    if policy.get("open_source_evaluation_is_preregistered_not_blinded") is not True:
        fail("open-source evaluation must be described as preregistered, not blinded")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    try:
        validate(args.manifest, args.source)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: MATH-REP-001D0 property-generator manifest is internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
