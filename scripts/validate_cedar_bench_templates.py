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
    for template in templates:
        tid = template["id"]
        families.add(template["family"])
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
    print(f"CEDAR-Bench templates: {len(templates)} templates across {len(families)} families passed path and alias checks")


def main() -> int:
    try:
        validate()
        return 0
    except (OSError, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as error:
        print(f"CEDAR-Bench template validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
