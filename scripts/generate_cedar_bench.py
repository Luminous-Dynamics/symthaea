#!/usr/bin/env python3
"""Expand CEDAR-Bench templates into isolated candidate and auditor JSONL streams.

The seed is supplied out of band via CEDAR_BENCH_SEED. It is never printed or
written to the candidate stream. This is a harness fixture generator, not a
semantic scorer or model runner.
"""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "docs/research/cedar-bench/CEDAR_BENCH_V0_TEMPLATES.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def keyed(seed: bytes, *parts: str) -> int:
    message = "|".join(parts).encode("utf-8")
    return int.from_bytes(hmac.new(seed, message, hashlib.sha256).digest()[:8], "big")


def derive_value(seed: bytes, template_id: str, fact: dict[str, Any], variant: str) -> str:
    kind = fact["kind"]
    n = keyed(seed, template_id, fact["id"], variant)
    if kind in {"integer", "decimal"}:
        if kind == "integer":
            lo, hi = int(fact["min"]), int(fact["max"])
            value = lo + (n % (hi - lo + 1))
            return str(value)
        step = float(fact["step"])
        lo_units = int(round(float(fact["min"]) / step))
        hi_units = int(round(float(fact["max"]) / step))
        units = lo_units + (n % (hi_units - lo_units + 1))
        decimals = max(0, len(str(fact["step"]).split(".")[-1]))
        return f"{units * step:.{decimals}f}"
    if kind == "enum":
        values = fact["values"]
        return str(values[n % len(values)])
    if kind == "date":
        start = dt.date.fromisoformat(fact["start"])
        offset = n % int(fact["days"])
        return (start + dt.timedelta(days=offset)).isoformat()
    raise ValueError(f"unsupported value kind: {kind}")


def distinct_value(seed: bytes, template_id: str, fact: dict[str, Any], baseline: str) -> str:
    for attempt in range(256):
        value = derive_value(seed, template_id, fact, f"counterfactual-{attempt}")
        if value != baseline:
            return value
    raise ValueError(f"could not derive distinct counterfactual for {template_id}/{fact['id']}")


def replace_markers(value: Any, markers: dict[str, str]) -> Any:
    if isinstance(value, str):
        for marker, replacement in markers.items():
            value = value.replace(marker, replacement)
        return value
    if isinstance(value, list):
        return [replace_markers(item, markers) for item in value]
    if isinstance(value, dict):
        return {key: replace_markers(item, markers) for key, item in value.items()}
    return value


def source_digest(source: dict[str, Any]) -> str:
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return digest(canonical)


def expand(seed: bytes, template: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    template_id = template["template_id"]
    facts = template["facts"]
    baseline_values = {f["id"]: derive_value(seed, template_id, f, "baseline") for f in facts}
    counter_values = dict(baseline_values)
    selected = next((f for f in facts if f.get("intervene", True)), facts[0])
    counter_values[selected["id"]] = distinct_value(seed, template_id, selected, baseline_values[selected["id"]])
    placebo_base = derive_value(seed, template_id, template["placebo"], "placebo-baseline")
    placebo_cf = distinct_value(seed, template_id, template["placebo"], placebo_base)

    conditions = []
    # Condition descriptors remain in auditor output only.
    for condition in ("baseline", "counterfactual", "placebo", "source_removed"):
        values = baseline_values if condition != "counterfactual" else counter_values
        placebo = placebo_cf if condition == "placebo" else placebo_base
        markers = {f["placeholder"]: values[f["id"]] for f in facts}
        markers[template["placebo"]["placeholder"]] = placebo
        bundle = replace_markers(copy.deepcopy(template["source"]), markers)
        if condition == "source_removed" and "source_removal" in template:
            remove_id = template["source_removal"]["remove_source_id"]
            bundle["sources"] = [s for s in bundle["sources"] if s["source_id"] != remove_id]
        elif condition == "source_removed":
            remove_id = bundle["sources"][0]["source_id"]
            bundle["sources"] = [s for s in bundle["sources"] if s["source_id"] != remove_id]
        digests = {s["source_id"]: source_digest(s) for s in bundle["sources"]}
        opaque = hmac.new(seed, f"{template_id}|{condition}|trial".encode(), hashlib.sha256).hexdigest()[:24]
        candidate = {
            "task_id": opaque,
            "prompt": template["prompt"],
            "sources": bundle["sources"],
            "candidate_output_contract": ["answer", "citations", "abstention_state", "source_conflict_state"],
        }
        expected = template["expected"]
        if condition == "counterfactual":
            expected_behavior = "TRACKS_COUNTERFACTUAL" if expected not in {"SOURCE_CONFLICT_REPRESENTED"} else expected
        elif condition == "placebo":
            expected_behavior = "PLACEBO_STABLE"
        elif condition == "source_removed":
            expected_behavior = template.get("source_removal", {}).get("expected", "ABSTAIN_OR_MARK_UNRESOLVED")
        else:
            expected_behavior = "BASELINE_GROUNDED"
        audit = {
            "task_id": opaque,
            "template_id": template_id,
            "condition": condition,
            "expanded_values": values,
            "placebo_baseline": placebo_base,
            "placebo_variant": placebo,
            "source_digests": digests,
            "expected_behavior": expected_behavior,
            "seed_commitment": digest(seed),
            "selected_intervention_fact": selected["id"],
        }
        conditions.append((candidate, audit))
    return [x[0] for x in conditions], [x[1] for x in conditions]


def main() -> int:
    try:
        raw_seed = os.environ.get("CEDAR_BENCH_SEED", "")
        seed = raw_seed.encode("utf-8")
        if len(seed) < 32:
            raise ValueError("CEDAR_BENCH_SEED must contain at least 32 bytes")
        candidate_out = Path(sys.argv[1])
        audit_out = Path(sys.argv[2])
        document = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        candidate_rows, audit_rows = [], []
        for template in document["templates"]:
            candidates, audits = expand(seed, template)
            candidate_rows.extend(candidates)
            audit_rows.extend(audits)
        # Shuffle paired conditions deterministically without revealing condition.
        order = sorted(range(len(candidate_rows)), key=lambda i: keyed(seed, str(i), "order"))
        candidate_out.parent.mkdir(parents=True, exist_ok=True)
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        candidate_out.write_text("".join(json.dumps(candidate_rows[i], sort_keys=True) + "\n" for i in order), encoding="utf-8")
        audit_out.write_text("".join(json.dumps(audit_rows[i], sort_keys=True) + "\n" for i in order), encoding="utf-8")
        print(f"generated {len(candidate_rows)} candidate trials; auditor key is written separately")
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"CEDAR-Bench expansion failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
