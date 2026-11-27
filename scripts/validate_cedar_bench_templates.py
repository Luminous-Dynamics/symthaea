#!/usr/bin/env python3
"""Validate CEDAR-Bench template definitions without exposing generated values."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/research/cedar-bench/CEDAR_BENCH_V0_TEMPLATES.json"
MARKER = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in collect_strings(item)]
    if isinstance(value, dict):
        return [s for item in value.values() for s in collect_strings(item)]
    return []


def validate() -> None:
    doc = json.loads(PATH.read_text(encoding="utf-8"))
    templates = doc["templates"]
    require(doc["benchmark"] == "CEDAR-Bench", "benchmark name mismatch")
    require(len(templates) == 20, f"expected 20 templates, found {len(templates)}")
    ids = [t["template_id"] for t in templates]
    require(len(set(ids)) == len(ids), "template IDs must be unique")
    required_conditions = set(doc["conditions"])
    families = set()
    for template in templates:
        families.add(template["family"])
        sources = template["source"]["sources"]
        require(bool(sources), f"{template['template_id']}: missing source")
        source_ids = [s["source_id"] for s in sources]
        require(len(source_ids) == len(set(source_ids)), f"{template['template_id']}: duplicate source IDs")
        strings = collect_strings(template["source"])
        facts = template["facts"]
        fact_ids = [f["id"] for f in facts]
        require(bool(facts) and len(fact_ids) == len(set(fact_ids)), f"{template['template_id']}: invalid facts")
        for fact in facts:
            marker = fact["placeholder"]
            require(strings and sum(s.count(marker) for s in strings) >= 2,
                    f"{template['template_id']}/{fact['id']}: fact marker needs at least two source aliases")
            require(bool(fact.get("aliases")), f"{template['template_id']}/{fact['id']}: no declared alias paths")
            kind = fact["kind"]
            require(kind in {"integer", "decimal", "enum", "date"}, f"{template['template_id']}: unsupported fact type")
            if kind == "decimal":
                require(float(fact["min"]) < float(fact["max"]) and float(fact["step"]) > 0,
                        f"{template['template_id']}: invalid decimal range")
            if kind == "integer":
                require(int(fact["min"]) < int(fact["max"]), f"{template['template_id']}: invalid integer range")
            if kind == "enum":
                require(len(set(fact["values"])) >= 2, f"{template['template_id']}: enum needs distinct values")
            if kind == "date":
                require(int(fact["days"]) > 1, f"{template['template_id']}: date interval too small")
        placebo = template["placebo"]
        require(sum(s.count(placebo["placeholder"]) for s in strings) >= 1,
                f"{template['template_id']}: placebo placeholder missing")
        if "source_removal" in template:
            require(template["source_removal"]["remove_source_id"] in source_ids,
                    f"{template['template_id']}: removal source absent")
        if "citation_swap" in template:
            require(template["citation_swap"]["decoy_source_id"] in source_ids,
                    f"{template['template_id']}: citation-swap decoy absent")
        require(template["expected"], f"{template['template_id']}: expected behavior label missing")
    require(len(families) >= 10, "need at least ten distinct evidence-dependence families")
    print(f"CEDAR-Bench templates: {len(templates)} templates across {len(families)} families passed schema/alias checks")


def main() -> int:
    try:
        validate()
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"CEDAR-Bench template validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
