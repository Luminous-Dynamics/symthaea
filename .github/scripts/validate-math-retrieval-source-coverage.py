#!/usr/bin/env python3
"""Validate common source/representation coverage for qualified math retrieval."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

VERSION = "math-retrieval-source-coverage-v1"
REPORT_VERSION = "math-retrieval-source-coverage-validation-report-v1"
AUTHORITY = "MeasurementOnly"
ROOT = {
    "version", "coverage_id", "authority", "candidate_universe", "parser",
    "normalizer", "coverage_policy", "targets", "rows",
}
CANDIDATE_UNIVERSE = {
    "corpus_snapshot_sha256", "knowledge_boundary_sha256",
    "candidate_eligibility_policy_sha256", "candidate_set_sha256",
    "candidate_count", "source_identity_kind", "canonical_candidate_order",
}
PARSER = {
    "source_object_contract_sha256", "parser_contract_sha256",
    "parser_implementation_sha256", "parsed_object_kind",
    "parsed_object_serialization_sha256",
}
NORMALIZER = {
    "normalization_contract_sha256", "normalization_implementation_sha256",
    "normalized_object_kind", "normalized_object_serialization_sha256",
}
TARGET = {
    "target_id", "channel", "representation_family", "representation_sha256",
    "item_serialization_sha256", "input_stage", "max_serialized_item_bytes",
}
ROW = {
    "source_object_sha256", "parse_status", "parsed_object_sha256",
    "normalization_status", "normalized_object_sha256", "representations",
}
REP = {"target_id", "status", "representation_object_sha256", "serialized_bytes"}


class ValidationError(ValueError):
    pass


def closed(obj: object, fields: set[str], where: str) -> dict:
    if not isinstance(obj, dict):
        raise ValidationError(f"{where}: object required")
    extra = set(obj) - fields
    missing = fields - set(obj)
    if extra:
        raise ValidationError(f"{where}: unknown fields {sorted(extra)}")
    if missing:
        raise ValidationError(f"{where}: missing fields {sorted(missing)}")
    return obj


def text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where}: non-empty string required")
    return value


def sha(value: object, where: str) -> str:
    value = text(value, where)
    if len(value) != 71 or not value.startswith("sha256:"):
        raise ValidationError(f"{where}: sha256:<64 lowercase hex> required")
    if any(ch not in "0123456789abcdef" for ch in value[7:]):
        raise ValidationError(f"{where}: invalid SHA-256")
    return value


def positive_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValidationError(f"{where}: positive integer required")
    return value


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load_sibling(filename: str, module_name: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_document(doc: object) -> dict[str, dict]:
    root = closed(doc, ROOT, "root")
    if root["version"] != VERSION or root["authority"] != AUTHORITY:
        raise ValidationError("root: version/authority invariant failed")
    text(root["coverage_id"], "coverage_id")
    if root["coverage_policy"] != "CommonIntersectionRequired":
        raise ValidationError("coverage_policy: CommonIntersectionRequired required")

    universe = closed(root["candidate_universe"], CANDIDATE_UNIVERSE, "candidate_universe")
    for field in (
        "corpus_snapshot_sha256", "knowledge_boundary_sha256",
        "candidate_eligibility_policy_sha256", "candidate_set_sha256",
    ):
        sha(universe[field], f"candidate_universe.{field}")
    count = positive_int(universe["candidate_count"], "candidate_universe.candidate_count")
    if universe["source_identity_kind"] != "SourceObjectDigest":
        raise ValidationError("candidate_universe.source_identity_kind: SourceObjectDigest required")
    if universe["canonical_candidate_order"] != "SourceObjectDigestAscending":
        raise ValidationError("candidate_universe.canonical_candidate_order: SourceObjectDigestAscending required")

    parser = closed(root["parser"], PARSER, "parser")
    for field in PARSER - {"parsed_object_kind"}:
        sha(parser[field], f"parser.{field}")
    if parser["parsed_object_kind"] != "FolFormulaExt":
        raise ValidationError("parser.parsed_object_kind: FolFormulaExt required")

    normalizer = closed(root["normalizer"], NORMALIZER, "normalizer")
    for field in NORMALIZER - {"normalized_object_kind"}:
        sha(normalizer[field], f"normalizer.{field}")
    if normalizer["normalized_object_kind"] != "ExactNormalForm":
        raise ValidationError("normalizer.normalized_object_kind: ExactNormalForm required")

    targets = root["targets"]
    if not isinstance(targets, list) or not 1 <= len(targets) <= 16:
        raise ValidationError("targets: 1..16 entries required")
    target_map: dict[str, dict] = {}
    ordered_target_ids: list[str] = []
    target_identity_keys: set[tuple[str, str, str]] = set()
    for i, raw in enumerate(targets):
        target = closed(raw, TARGET, f"targets[{i}]")
        target_id = text(target["target_id"], f"targets[{i}].target_id")
        if len(target_id) > 64 or not target_id[0].isalpha() or any(
            not (ch.isalnum() or ch in "._-") for ch in target_id
        ):
            raise ValidationError(f"targets[{i}].target_id: invalid identifier")
        if target_id in target_map:
            raise ValidationError("targets: duplicate target_id")
        channel = target["channel"]
        family = target["representation_family"]
        stage = target["input_stage"]
        if channel not in {"Syntax", "ExactNormalForm"}:
            raise ValidationError(f"targets[{i}].channel: unsupported")
        if family not in {"Lexical", "CanonicalSparse", "HDC"}:
            raise ValidationError(f"targets[{i}].representation_family: unsupported")
        if stage not in {"SourceObject", "ParsedFolFormulaExt", "ExactNormalForm"}:
            raise ValidationError(f"targets[{i}].input_stage: unsupported")
        if stage == "SourceObject" and not (channel == "Syntax" and family == "Lexical"):
            raise ValidationError(f"targets[{i}]: SourceObject input is reserved for Syntax/Lexical")
        if stage == "ParsedFolFormulaExt" and channel != "Syntax":
            raise ValidationError(f"targets[{i}]: ParsedFolFormulaExt requires Syntax channel")
        if stage == "ExactNormalForm" and channel != "ExactNormalForm":
            raise ValidationError(f"targets[{i}]: ExactNormalForm input requires ExactNormalForm channel")
        if channel == "ExactNormalForm" and stage != "ExactNormalForm":
            raise ValidationError(f"targets[{i}]: ExactNormalForm channel must consume ExactNormalForm")
        sha(target["representation_sha256"], f"targets[{i}].representation_sha256")
        sha(target["item_serialization_sha256"], f"targets[{i}].item_serialization_sha256")
        positive_int(target["max_serialized_item_bytes"], f"targets[{i}].max_serialized_item_bytes")
        identity_key = (channel, family, target["representation_sha256"])
        if identity_key in target_identity_keys:
            raise ValidationError("targets: duplicate channel/family/representation identity")
        target_identity_keys.add(identity_key)
        target_map[target_id] = target
        ordered_target_ids.append(target_id)
    if ordered_target_ids != sorted(ordered_target_ids):
        raise ValidationError("targets: canonical target_id ascending order required")

    rows = root["rows"]
    if not isinstance(rows, list) or len(rows) != count:
        raise ValidationError("rows: must contain exactly candidate_count entries")
    source_ids: list[str] = []
    representation_occurrences = 0
    for i, raw in enumerate(rows):
        row = closed(raw, ROW, f"rows[{i}]")
        source = sha(row["source_object_sha256"], f"rows[{i}].source_object_sha256")
        source_ids.append(source)
        if row["parse_status"] != "Parsed":
            raise ValidationError(f"rows[{i}].parse_status: Parsed required by common-intersection policy")
        sha(row["parsed_object_sha256"], f"rows[{i}].parsed_object_sha256")
        if row["normalization_status"] != "Normalized":
            raise ValidationError(f"rows[{i}].normalization_status: Normalized required by common-intersection policy")
        sha(row["normalized_object_sha256"], f"rows[{i}].normalized_object_sha256")
        reps = row["representations"]
        if not isinstance(reps, list) or len(reps) != len(targets):
            raise ValidationError(f"rows[{i}].representations: exactly one entry per target required")
        rep_target_ids: list[str] = []
        for j, raw_rep in enumerate(reps):
            rep = closed(raw_rep, REP, f"rows[{i}].representations[{j}]")
            target_id = text(rep["target_id"], f"rows[{i}].representations[{j}].target_id")
            rep_target_ids.append(target_id)
            if target_id not in target_map:
                raise ValidationError(f"rows[{i}].representations[{j}]: unknown target_id")
            if rep["status"] != "Ready":
                raise ValidationError(f"rows[{i}].representations[{j}].status: Ready required")
            sha(rep["representation_object_sha256"], f"rows[{i}].representations[{j}].representation_object_sha256")
            size = positive_int(rep["serialized_bytes"], f"rows[{i}].representations[{j}].serialized_bytes")
            if size > target_map[target_id]["max_serialized_item_bytes"]:
                raise ValidationError(
                    f"rows[{i}].representations[{j}]: serialized bytes exceed frozen target ceiling"
                )
            representation_occurrences += 1
        if rep_target_ids != ordered_target_ids:
            raise ValidationError(f"rows[{i}].representations: target order/set must exactly match targets")
    if source_ids != sorted(source_ids):
        raise ValidationError("rows: SourceObjectDigestAscending order required")
    if len(set(source_ids)) != len(source_ids):
        raise ValidationError("rows: duplicate source_object_sha256")
    return {
        "universe": universe,
        "parser": parser,
        "normalizer": normalizer,
        "target_map": target_map,
        "source_ids": source_ids,
        "representation_occurrences": representation_occurrences,
    }


def validate_bound(
    coverage_doc: object,
    coverage_raw: bytes,
    candidate_doc: object,
    candidate_raw: bytes,
    experiment_doc: object,
) -> dict:
    state = validate_document(coverage_doc)
    universe = state["universe"]

    candidate_mod = load_sibling(
        "validate-math-retrieval-candidate-set.py", "sym_source_coverage_candidate_validator"
    )
    candidate_mod.validate(candidate_doc)
    candidate_sha = digest_bytes(candidate_raw)
    if candidate_sha != universe["candidate_set_sha256"]:
        raise ValidationError("candidate-set exact bytes do not match candidate_universe.candidate_set_sha256")
    for field in (
        "corpus_snapshot_sha256", "knowledge_boundary_sha256",
        "candidate_eligibility_policy_sha256", "candidate_count",
    ):
        if candidate_doc[field] != universe[field]:
            raise ValidationError(f"candidate-set {field} differs from coverage universe")
    if candidate_doc["source_identity_kind"] != universe["source_identity_kind"]:
        raise ValidationError("candidate-set source_identity_kind differs from coverage universe")
    if candidate_doc["canonical_order"] != universe["canonical_candidate_order"]:
        raise ValidationError("candidate-set canonical order differs from coverage universe")
    if candidate_doc["candidates"] != state["source_ids"]:
        raise ValidationError("coverage rows do not exactly equal the frozen candidate set")

    exp_mod = load_sibling(
        "validate-math-search-experiment-v2.1.py", "sym_source_coverage_experiment_validator"
    )
    exp_mod.validate(experiment_doc)
    shared = experiment_doc["shared_contract"]
    if universe["corpus_snapshot_sha256"] != shared["corpus_snapshot_sha256"]:
        raise ValidationError("coverage corpus snapshot differs from experiment shared contract")
    if universe["knowledge_boundary_sha256"] != shared["knowledge_boundary_sha256"]:
        raise ValidationError("coverage knowledge boundary differs from experiment shared contract")
    if state["parser"]["source_object_contract_sha256"] != shared["source_object_contract_sha256"]:
        raise ValidationError("coverage source-object contract differs from experiment shared contract")
    if state["normalizer"]["normalization_contract_sha256"] != shared["normalization_contract_sha256"]:
        raise ValidationError("coverage normalization contract differs from experiment shared contract")
    if state["normalizer"]["normalization_implementation_sha256"] != shared["normalization_implementation_sha256"]:
        raise ValidationError("coverage normalization implementation differs from experiment shared contract")

    return {
        "version": REPORT_VERSION,
        "authority": AUTHORITY,
        "coverage_sha256": digest_bytes(coverage_raw),
        "candidate_set_sha256": candidate_sha,
        "candidate_count": universe["candidate_count"],
        "target_count": len(state["target_map"]),
        "representation_occurrences": state["representation_occurrences"],
        "all_candidate_sources_covered": True,
        "all_required_targets_ready": True,
        "all_checks_passed": True,
    }


def d(n: int) -> str:
    return f"sha256:{n:064x}"


def fixture() -> tuple[dict, dict, dict]:
    candidate_mod = load_sibling(
        "validate-math-retrieval-candidate-set.py", "sym_source_coverage_candidate_fixture"
    )
    exp_mod = load_sibling(
        "validate-math-search-experiment-v2.1.py", "sym_source_coverage_experiment_fixture"
    )
    candidate = candidate_mod.fixture()
    experiment = exp_mod.fixture()
    candidate["corpus_snapshot_sha256"] = experiment["shared_contract"]["corpus_snapshot_sha256"]
    candidate["knowledge_boundary_sha256"] = experiment["shared_contract"]["knowledge_boundary_sha256"]
    candidate["candidate_eligibility_policy_sha256"] = d(500)
    candidate_raw = (json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n").encode()
    targets = [
        {
            "target_id": "H",
            "channel": "Syntax",
            "representation_family": "HDC",
            "representation_sha256": d(600),
            "item_serialization_sha256": d(601),
            "input_stage": "ParsedFolFormulaExt",
            "max_serialized_item_bytes": 4096,
        },
        {
            "target_id": "L",
            "channel": "Syntax",
            "representation_family": "Lexical",
            "representation_sha256": d(610),
            "item_serialization_sha256": d(611),
            "input_stage": "SourceObject",
            "max_serialized_item_bytes": 4096,
        },
        {
            "target_id": "N",
            "channel": "ExactNormalForm",
            "representation_family": "CanonicalSparse",
            "representation_sha256": d(620),
            "item_serialization_sha256": d(621),
            "input_stage": "ExactNormalForm",
            "max_serialized_item_bytes": 4096,
        },
        {
            "target_id": "S",
            "channel": "Syntax",
            "representation_family": "CanonicalSparse",
            "representation_sha256": d(630),
            "item_serialization_sha256": d(631),
            "input_stage": "ParsedFolFormulaExt",
            "max_serialized_item_bytes": 4096,
        },
    ]
    coverage = {
        "version": VERSION,
        "coverage_id": "fixture",
        "authority": AUTHORITY,
        "candidate_universe": {
            "corpus_snapshot_sha256": candidate["corpus_snapshot_sha256"],
            "knowledge_boundary_sha256": candidate["knowledge_boundary_sha256"],
            "candidate_eligibility_policy_sha256": candidate["candidate_eligibility_policy_sha256"],
            "candidate_set_sha256": digest_bytes(candidate_raw),
            "candidate_count": candidate["candidate_count"],
            "source_identity_kind": "SourceObjectDigest",
            "canonical_candidate_order": "SourceObjectDigestAscending",
        },
        "parser": {
            "source_object_contract_sha256": experiment["shared_contract"]["source_object_contract_sha256"],
            "parser_contract_sha256": d(700),
            "parser_implementation_sha256": d(701),
            "parsed_object_kind": "FolFormulaExt",
            "parsed_object_serialization_sha256": d(702),
        },
        "normalizer": {
            "normalization_contract_sha256": experiment["shared_contract"]["normalization_contract_sha256"],
            "normalization_implementation_sha256": experiment["shared_contract"]["normalization_implementation_sha256"],
            "normalized_object_kind": "ExactNormalForm",
            "normalized_object_serialization_sha256": d(703),
        },
        "coverage_policy": "CommonIntersectionRequired",
        "targets": targets,
        "rows": [],
    }
    for i, source in enumerate(candidate["candidates"]):
        coverage["rows"].append({
            "source_object_sha256": source,
            "parse_status": "Parsed",
            "parsed_object_sha256": d(800 + i),
            "normalization_status": "Normalized",
            "normalized_object_sha256": d(900 + i),
            "representations": [
                {
                    "target_id": target["target_id"],
                    "status": "Ready",
                    "representation_object_sha256": d(1000 + i * 10 + j),
                    "serialized_bytes": 128 + j,
                }
                for j, target in enumerate(targets)
            ],
        })
    return coverage, candidate, experiment


def self_test() -> None:
    coverage, candidate, experiment = fixture()
    candidate_raw = (json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n").encode()
    coverage_raw = (json.dumps(coverage, sort_keys=True, separators=(",", ":")) + "\n").encode()
    validate_bound(coverage, coverage_raw, candidate, candidate_raw, experiment)

    attacks = []
    def omit_source(c): c["rows"].pop()
    attacks.append(omit_source)
    def reorder_source(c): c["rows"].reverse()
    attacks.append(reorder_source)
    def duplicate_source(c): c["rows"][1]["source_object_sha256"] = c["rows"][0]["source_object_sha256"]
    attacks.append(duplicate_source)
    def silent_target_drop(c): c["rows"][0]["representations"].pop()
    attacks.append(silent_target_drop)
    def not_ready(c): c["rows"][0]["representations"][0]["status"] = "Unsupported"
    attacks.append(not_ready)
    def oversize(c): c["rows"][0]["representations"][0]["serialized_bytes"] = 999999
    attacks.append(oversize)
    def wrong_candidate_digest(c): c["candidate_universe"]["candidate_set_sha256"] = d(9999)
    attacks.append(wrong_candidate_digest)
    def wrong_normalizer(c): c["normalizer"]["normalization_implementation_sha256"] = d(9998)
    attacks.append(wrong_normalizer)
    def swap_target_order(c): c["targets"][0], c["targets"][1] = c["targets"][1], c["targets"][0]
    attacks.append(swap_target_order)

    for mutate in attacks:
        bad = copy.deepcopy(coverage)
        mutate(bad)
        raw = (json.dumps(bad, sort_keys=True, separators=(",", ":")) + "\n").encode()
        try:
            validate_bound(bad, raw, candidate, candidate_raw, experiment)
        except (ValidationError, ValueError):
            continue
        raise AssertionError(f"adversarial coverage self-test unexpectedly passed: {mutate.__name__}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage", nargs="?", type=Path)
    parser.add_argument("candidate_set", nargs="?", type=Path)
    parser.add_argument("experiment", nargs="?", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("math-retrieval source coverage v1 self-test: PASS")
        return 0
    if args.coverage is None or args.candidate_set is None or args.experiment is None:
        parser.error("coverage, candidate_set, and experiment required unless --self-test")
    try:
        coverage_raw = args.coverage.read_bytes()
        candidate_raw = args.candidate_set.read_bytes()
        coverage_doc = json.loads(coverage_raw.decode("utf-8"))
        candidate_doc = json.loads(candidate_raw.decode("utf-8"))
        experiment_doc = json.loads(args.experiment.read_text(encoding="utf-8"))
        report = validate_bound(
            coverage_doc, coverage_raw, candidate_doc, candidate_raw, experiment_doc
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    out = json.dumps(report, sort_keys=True, separators=(",", ":"))
    if args.report:
        args.report.write_text(out + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
