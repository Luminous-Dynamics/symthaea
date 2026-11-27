#!/usr/bin/env python3
"""Validate CEDAR-Bench paths, aliases and secret-seeded stream separation."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "docs/research/cedar-bench/CEDAR_BENCH_V0_TEMPLATES.json"
MARKER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
SENTINEL = "__CEDAR_FACT__"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def pointer_parts(pointer: str) -> list[str]:
    require(pointer == "" or pointer.startswith("/"), f"invalid JSON Pointer {pointer!r}")
    return [] if pointer == "" else [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def resolve(document: Any, pointer: str) -> Any:
    current = document
    for part in pointer_parts(pointer):
        if isinstance(current, list):
            require(part.isdigit() and int(part) < len(current), f"invalid array index at {pointer}")
            current = current[int(part)]
        else:
            require(isinstance(current, dict) and part in current, f"missing pointer path {pointer}")
            current = current[part]
    return current


def collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in collect_strings(item)]
    if isinstance(value, dict):
        return [s for item in value.values() for s in collect_strings(item)]
    return []


def validate() -> None:
    doc = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    templates = doc["templates"]
    require(doc["benchmark"] == "CEDAR-Bench", "benchmark name mismatch")
    require(len(templates) == 20, f"expected 20 templates, found {len(templates)}")
    ids = [template["id"] for template in templates]
    require(len(set(ids)) == len(ids), "template IDs must be unique")
    families = set()
    family_splits: dict[str, str] = {}
    split_counts = {"train": 0, "development": 0, "sealed_test": 0}
    for template in templates:
        tid = template["id"]
        families.add(template["family"])
        split = template.get("split")
        require(split in split_counts, f"{tid}: missing or invalid train/development/sealed_test split")
        require(template["family"] not in family_splits, f"{tid}: template family appears more than once")
        family_splits[template["family"]] = split
        split_counts[split] += 1
        source_bundle = template["source"]
        sources = source_bundle["sources"]
        require(bool(sources), f"{tid}: missing source")
        source_ids = [source["source_id"] for source in sources]
        require(len(source_ids) == len(set(source_ids)), f"{tid}: duplicate source IDs")
        fact_ids = [fact["id"] for fact in template["facts"]]
        require(bool(fact_ids) and len(fact_ids) == len(set(fact_ids)), f"{tid}: duplicate/missing fact IDs")
        for fact in template["facts"]:
            fact_id, marker = fact["id"], fact["placeholder"]
            require(marker == "{{" + fact_id + "}}", f"{tid}/{fact_id}: placeholder/id mismatch")
            require(bool(fact.get("aliases")), f"{tid}/{fact_id}: no declared alias paths")
            for pointer in fact["aliases"]:
                value = resolve(source_bundle, pointer)
                if isinstance(value, str):
                    require(marker in value, f"{tid}/{fact_id}: alias {pointer} lacks its marker")
                else:
                    require(value == {SENTINEL: fact_id},
                            f"{tid}/{fact_id}: typed JSON alias {pointer} is not bound to fact")
            kind = fact["kind"]
            require(kind in {"integer", "decimal", "enum", "date"}, f"{tid}: unsupported fact kind")
            if kind == "decimal":
                require(float(fact["min"]) < float(fact["max"]) and float(fact["step"]) > 0,
                        f"{tid}: invalid decimal range")
            if kind == "integer":
                require(int(fact["min"]) < int(fact["max"]), f"{tid}: invalid integer range")
            if kind == "enum":
                require(len(set(fact["values"])) >= 2, f"{tid}: enum needs distinct values")
            if kind == "date":
                require(int(fact["days"]) > 1, f"{tid}: date interval too small")
        answer_spec = template["answer_spec"]
        valid_fact_ids = set(fact_ids)
        kind = answer_spec["kind"]
        referenced_facts = set()
        if kind == "fact":
            referenced_facts.add(answer_spec["fact_id"])
        elif kind == "value_and_unit":
            referenced_facts.update([answer_spec["value_fact"], answer_spec["unit_fact"]])
        elif kind == "absolute_difference":
            referenced_facts.update([answer_spec["left_fact"], answer_spec["right_fact"]])
        elif kind == "ratio":
            referenced_facts.update([answer_spec["numerator_fact"], answer_spec["denominator_fact"]])
        elif kind == "fact_map":
            referenced_facts.update(answer_spec["fact_source_map"].keys())
        else:
            raise ValueError(f"{tid}: unknown answer-oracle kind {kind}")
        require(referenced_facts <= valid_fact_ids, f"{tid}: answer oracle references undeclared facts")
        referenced_sources = set(answer_spec.get("source_ids", []))
        if kind == "fact_map":
            referenced_sources.update(answer_spec["fact_source_map"].values())
        require(referenced_sources <= set(source_ids), f"{tid}: answer oracle references unknown sources")
        if kind == "fact" and answer_spec.get("source_mode", "all") not in {"all", "any"}:
            raise ValueError(f"{tid}: invalid source_mode")
        citation = template["citation_contract"]
        support_ids = set(citation["support_source_ids"])
        decoy_ids = set(citation["decoy_source_ids"])
        require(support_ids <= set(source_ids), f"{tid}: citation support source missing")
        require(referenced_sources <= support_ids, f"{tid}: answer oracle uses a source not allowed as support")
        require(decoy_ids <= set(source_ids), f"{tid}: citation decoy source missing")
        require(not (support_ids & decoy_ids), f"{tid}: a source cannot be both support and decoy")
        require(citation.get("must_not_cite_decoy") is True, f"{tid}: decoy-citation prohibition missing")
        placebo = template["placebo"]
        require(placebo.get("id") == "P", f"{tid}: placebo fact ID must be P")
        require(bool(placebo.get("paths")), f"{tid}: placebo paths missing")
        require(bool(re.fullmatch(r"\{\{[A-Z][A-Z0-9_]*\}\}", placebo["placeholder"])),
                f"{tid}: placebo placeholder invalid")
        for pointer in placebo["paths"]:
            parts = pointer_parts(pointer)
            require(len(parts) >= 3 and parts[0] == "sources", f"{tid}: placebo path must be under a source")
            require(parts[1].isdigit() and int(parts[1]) < len(sources), f"{tid}: placebo source index missing")
        if "source_removal" in template:
            require(template["source_removal"]["remove_source_id"] in source_ids,
                    f"{tid}: source-removal target missing")
        if "citation_swap" in template:
            require(template["citation_swap"]["decoy_source_id"] in source_ids,
                    f"{tid}: citation-swap decoy missing")
        strings = collect_strings(source_bundle)
        all_markers = [marker for text in strings for marker in MARKER.findall(text)]
        valid_marker_ids = set(fact_ids) | {placebo["placeholder"][2:-2]}
        require(set(all_markers) <= valid_marker_ids, f"{tid}: undeclared marker found")
        require(template["expected"], f"{tid}: missing expected-behavior contract")
    require(len(families) >= 10, "need at least ten distinct task families")
    require(split_counts == {"train": 10, "development": 5, "sealed_test": 5},
            f"unexpected split allocation: {split_counts}")
    require(doc["split_protocol"].get("split_unit") == "template_family",
            "splits must be assigned by family to prevent family leakage")
    require(doc["split_protocol"].get("no_template_family_leakage") is True,
            "template-family leakage control must be explicit")
    print(f"CEDAR-Bench templates: {len(templates)} templates across {len(families)} families; split counts {split_counts}; path and alias checks passed")


def main() -> int:
    try:
        validate()
        return 0
    except (OSError, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as error:
        print(f"CEDAR-Bench template validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
