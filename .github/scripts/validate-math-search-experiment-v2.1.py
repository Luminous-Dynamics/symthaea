#!/usr/bin/env python3
"""Validate Symthaea math-search experiment manifest v2.1.

v2.1 is a narrow successor to v2: it replaces the ambiguous arm-level
index_manifest_sha256 edge with retrieval_binding_sha256. All inherited causal,
budget, endpoint, control, memory, and contrast invariants are delegated to the
exact sibling v2 validator after a lossless compatibility projection.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "symthaea.math-search-experiment.v2.1"
PARENT_SCHEMA_VERSION = "symthaea.math-search-experiment.v2"
BINDING_FIELD = "retrieval_binding_sha256"
LEGACY_FIELD = "index_manifest_sha256"


class ValidationError(ValueError):
    pass


def _load_parent():
    path = Path(__file__).with_name("validate-math-search-experiment-v2.py")
    spec = importlib.util.spec_from_file_location("symthaea_math_exp_v2_validator", path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load parent validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding_shape(doc: object) -> None:
    if not isinstance(doc, dict):
        raise ValidationError("root: expected object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"root.schema_version: expected {SCHEMA_VERSION}")
    arms = doc.get("arms")
    if not isinstance(arms, list):
        raise ValidationError("root.arms: expected list")

    for index, arm in enumerate(arms):
        where = f"root.arms[{index}]"
        if not isinstance(arm, dict):
            raise ValidationError(f"{where}: expected object")
        if LEGACY_FIELD in arm:
            raise ValidationError(f"{where}.{LEGACY_FIELD}: forbidden in v2.1; use {BINDING_FIELD}")
        retriever = arm.get("retriever_family")
        if retriever == "None":
            if BINDING_FIELD in arm:
                raise ValidationError(f"{where}: no-retrieval arm must not carry {BINDING_FIELD}")
        else:
            value = arm.get(BINDING_FIELD)
            if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
                raise ValidationError(f"{where}.{BINDING_FIELD}: sha256:<64 lowercase hex> required")
            if any(ch not in "0123456789abcdef" for ch in value[7:]):
                raise ValidationError(f"{where}.{BINDING_FIELD}: invalid SHA-256")


def _project_to_v2(doc: dict) -> dict:
    projected = copy.deepcopy(doc)
    projected["schema_version"] = PARENT_SCHEMA_VERSION
    for arm in projected["arms"]:
        if BINDING_FIELD in arm:
            arm[LEGACY_FIELD] = arm.pop(BINDING_FIELD)
    return projected


def validate_manifest(doc: object) -> None:
    _binding_shape(doc)
    parent = _load_parent()
    projected = _project_to_v2(doc)
    try:
        parent.validate(projected)
    except Exception as exc:
        parent_error = getattr(parent, "V", ValueError)
        if isinstance(exc, parent_error):
            raise ValidationError(str(exc)) from exc
        raise


def fixture() -> dict:
    parent = _load_parent()
    doc = parent.fixture()
    doc["schema_version"] = SCHEMA_VERSION
    for arm in doc["arms"]:
        if LEGACY_FIELD in arm:
            arm[BINDING_FIELD] = arm.pop(LEGACY_FIELD)
    return doc


def self_test() -> None:
    valid = fixture()
    validate_manifest(valid)

    attacks = []

    def add(name, mutate):
        attacks.append((name, mutate))

    add("retrieval arm missing binding", lambda d: d["arms"][3].pop(BINDING_FIELD))
    add("baseline smuggles binding", lambda d: d["arms"][0].__setitem__(BINDING_FIELD, "sha256:" + "0" * 64))
    add("legacy singular index field", lambda d: d["arms"][3].__setitem__(LEGACY_FIELD, d["arms"][3][BINDING_FIELD]))
    add("malformed binding digest", lambda d: d["arms"][3].__setitem__(BINDING_FIELD, "sha256:bad"))
    add("inherited budget attack", lambda d: d["arms"][3].__setitem__("retrieved_items_max", 999))
    add("inherited fusion-policy removal", lambda d: d["arms"][6].pop("fusion_policy_sha256"))

    for name, mutate in attacks:
        candidate = copy.deepcopy(valid)
        mutate(candidate)
        try:
            validate_manifest(candidate)
        except ValidationError:
            continue
        raise AssertionError(f"self-test attack unexpectedly passed: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("math-search experiment v2.1 self-test: PASS")
        return 0
    if args.path is None:
        parser.error("path is required unless --self-test is used")

    try:
        validate_manifest(json.loads(args.path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
