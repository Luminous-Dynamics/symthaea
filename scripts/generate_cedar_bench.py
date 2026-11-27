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
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "docs/research/cedar-bench/CEDAR_BENCH_V0_TEMPLATES.json"
FACT_SENTINEL = "__CEDAR_FACT__"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def keyed(seed: bytes, *parts: str) -> int:
    message = "|".join(parts).encode("utf-8")
    return int.from_bytes(hmac.new(seed, message, hashlib.sha256).digest()[:8], "big")


def derive_value(seed: bytes, template_id: str, fact: dict[str, Any], variant: str) -> str:
    kind = fact["kind"]
    n = keyed(seed, template_id, fact["id"], variant)
    if kind == "integer":
        lo, hi = int(fact["min"]), int(fact["max"])
        return str(lo + (n % (hi - lo + 1)))
    if kind == "decimal":
        step = Decimal(str(fact["step"]))
        lo_units = int(Decimal(str(fact["min"])) / step)
        hi_units = int(Decimal(str(fact["max"])) / step)
        units = lo_units + (n % (hi_units - lo_units + 1))
        return format(Decimal(units) * step, "f")
    if kind == "enum":
        values = fact["values"]
        return str(values[n % len(values)])
    if kind == "date":
        start = dt.date.fromisoformat(fact["start"])
        return (start + dt.timedelta(days=n % int(fact["days"]))).isoformat()
    raise ValueError(f"unsupported value kind: {kind}")


def distinct_value(seed: bytes, template_id: str, fact: dict[str, Any], baseline: str) -> str:
    for attempt in range(256):
        value = derive_value(seed, template_id, fact, f"counterfactual-{attempt}")
        if value != baseline:
            return value
    raise ValueError(f"could not derive distinct counterfactual for {template_id}/{fact['id']}")


def replace_markers(value: Any, markers: dict[str, str], fact_kinds: dict[str, str]) -> Any:
    if isinstance(value, dict):
        if set(value) == {FACT_SENTINEL}:
            fact_id = value[FACT_SENTINEL]
            require(fact_id in markers, f"unknown typed fact placeholder: {fact_id}")
            lexeme = markers[fact_id]
            if fact_kinds[fact_id] == "integer":
                return int(lexeme)
            if fact_kinds[fact_id] == "decimal":
                return float(Decimal(lexeme))
            return lexeme
        return {key: replace_markers(item, markers, fact_kinds) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_markers(item, markers, fact_kinds) for item in value]
    if isinstance(value, str):
        for fact_id, lexeme in markers.items():
            value = value.replace("{{" + fact_id + "}}", lexeme)
        return value
    return value


def pointer_parts(pointer: str) -> list[str]:
    require(pointer == "" or pointer.startswith("/"), f"invalid JSON Pointer: {pointer}")
    if not pointer:
        return []
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def set_pointer(document: Any, pointer: str, value: Any) -> None:
    tokens = pointer_parts(pointer)
    require(bool(tokens), "cannot replace root source bundle")
    current = document
    for index, token in enumerate(tokens[:-1]):
        next_token = tokens[index + 1]
        if isinstance(current, list):
            require(token.isdigit() and int(token) < len(current), f"invalid array index in {pointer}")
            current = current[int(token)]
        else:
            require(isinstance(current, dict), f"non-object parent in {pointer}")
            if token not in current:
                current[token] = [] if next_token.isdigit() else {}
            current = current[token]
    final = tokens[-1]
    if isinstance(current, list):
        require(final.isdigit() and int(final) < len(current), f"invalid terminal array index in {pointer}")
        current[int(final)] = value
    else:
        require(isinstance(current, dict), f"non-object terminal parent in {pointer}")
        current[final] = value


def source_digest(source: dict[str, Any]) -> str:
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return digest(canonical)


def bundle_digest(sources: list[dict[str, Any]]) -> str:
    canonical = json.dumps(sources, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return digest(canonical)


def expand(seed: bytes, template: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    template_id = template["id"]
    facts = template["facts"]
    baseline_values = {fact["id"]: derive_value(seed, template_id, fact, "baseline") for fact in facts}
    counter_values = dict(baseline_values)
    selected = next((fact for fact in facts if fact.get("intervene", True)), facts[0])
    counter_values[selected["id"]] = distinct_value(seed, template_id, selected, baseline_values[selected["id"]])
    fact_kinds = {fact["id"]: fact["kind"] for fact in facts}
    placebo_fact = dict(template["placebo"], id=template["placebo"].get("id", "P"))
    placebo_base = derive_value(seed, template_id, placebo_fact, "placebo-baseline")
    placebo_counterfactual = distinct_value(seed, template_id, placebo_fact, placebo_base)

    candidate_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for condition in ("baseline", "counterfactual", "placebo", "source_removed"):
        values = baseline_values if condition != "counterfactual" else counter_values
        placebo_value = placebo_counterfactual if condition == "placebo" else placebo_base
        expanded = replace_markers(copy.deepcopy(template["source"]), values, fact_kinds)
        for pointer in template["placebo"]["paths"]:
            prefix = template["placebo"].get("prefix", "")
            set_pointer(expanded, pointer, f"{prefix}{placebo_value}")
        sources = expanded["sources"]
        removed_source = None
        if condition == "source_removed":
            remove_id = template.get("source_removal", {}).get("remove_source_id", sources[0]["source_id"])
            removed_source = remove_id
            sources = [source for source in sources if source["source_id"] != remove_id]
        source_hashes = {source["source_id"]: source_digest(source) for source in sources}
        bundle_hash = bundle_digest(sources)
        opaque_id = hmac.new(seed, f"{template_id}|{condition}|trial".encode(), hashlib.sha256).hexdigest()[:24]
        candidate_rows.append({
            "task_id": opaque_id,
            "prompt": template["prompt"],
            "sources": sources,
            "candidate_output_contract": ["answer", "citations", "abstention_state", "source_conflict_state"],
        })
        if condition == "counterfactual":
            behavior = "TRACKS_COUNTERFACTUAL" if template["expected"] != "SOURCE_CONFLICT_REPRESENTED" else template["expected"]
        elif condition == "placebo":
            behavior = "PLACEBO_STABLE"
        elif condition == "source_removed":
            behavior = template.get("source_removal", {}).get("expected", "ABSTAIN_OR_MARK_UNRESOLVED")
        else:
            behavior = "BASELINE_GROUNDED"
        audit_rows.append({
            "task_id": opaque_id,
            "template_id": template_id,
            "condition": condition,
            "split": template["split"],
            "baseline_values": baseline_values,
            "counterfactual_values": counter_values,
            "expanded_values": values,
            "placebo_baseline": placebo_base,
            "placebo_variant": placebo_value,
            "removed_source_id": removed_source,
            "source_digests": source_hashes,
            "source_bundle_sha256": bundle_hash,
            "expected_behavior": behavior,
            "seed_commitment": digest(seed),
            "selected_intervention_fact": selected["id"],
        })
    return candidate_rows, audit_rows


def main() -> int:
    try:
        seed = os.environ.get("CEDAR_BENCH_SEED", "").encode("utf-8")
        require(len(seed) >= 32 and len(set(seed)) > 8,
                "CEDAR_BENCH_SEED must be high entropy and contain at least 32 bytes")
        candidate_out, audit_out = Path(sys.argv[1]), Path(sys.argv[2])
        require(candidate_out.resolve() != audit_out.resolve(), "candidate and auditor output paths must differ")
        document = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        candidate_rows, audit_rows = [], []
        requested_split = sys.argv[3] if len(sys.argv) > 3 else "sealed_test"
        require(requested_split in {"train", "development", "sealed_test", "all"},
                "split must be train, development, sealed_test or all")
        selected_templates = [
            template for template in document["templates"]
            if requested_split == "all" or template["split"] == requested_split
        ]
        require(bool(selected_templates), f"no templates assigned to split {requested_split}")
        for template in selected_templates:
            candidates, audits = expand(seed, template)
            candidate_rows.extend(candidates)
            audit_rows.extend(audits)
        require(len(candidate_rows) == len(audit_rows) == len(selected_templates) * 4,
                "candidate/auditor trial count mismatch")
        order = sorted(range(len(candidate_rows)), key=lambda index: keyed(seed, str(index), "order"))
        candidate_out.parent.mkdir(parents=True, exist_ok=True)
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        candidate_out.write_text("".join(json.dumps(candidate_rows[i], sort_keys=True) + "\n" for i in order), encoding="utf-8")
        audit_out.write_text("".join(json.dumps(audit_rows[i], sort_keys=True) + "\n" for i in order), encoding="utf-8")
        print(f"generated {len(candidate_rows)} {requested_split} candidate trials; auditor key is written separately")
        return 0
    except (OSError, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as error:
        print(f"CEDAR-Bench expansion failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
