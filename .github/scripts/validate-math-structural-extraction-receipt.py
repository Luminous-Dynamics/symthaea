#!/usr/bin/env python3
"""Validate the frozen MATH-RET-INTEGRATION-001 extraction subject/receipt.

Stdlib-only by design. The JSON schema is documentation/interchange; this script
is the semantic fail-closed gate used by CI.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED = {
    "version": "math-structural-extraction-subject-v1",
    "authority": "MeasurementOnly",
    "preparation_base_commit": "a884f7770f7df98a9a955881cf03e813d709d7de",
    "hdc_commit": "d52e08164ce44201dea3fa56b78dfb93dd35cfd3",
    "hdc_blob": "35ab3f0d51dc7298d80836f5efec2878a73bb75d",
    "hdc_path": "crates/core/symthaea-core/examples/math_structural_hdc_q0.rs",
    "hdc_id": "symthaea-math-structural-hdc-v1",
    "ast_commit": "1f38f8d0a217df7fd01715602058234c85cfe58c",
    "ast_blob": "1b1253b976a0a918d149b0acbb0a85e65f9f1f39",
    "ast_path": "crates/core/symthaea-core/examples/math_structural_retrieval_q0.rs",
    "ast_id": "canonical-ast-sparse-v1",
    "dev_blob": "491ffb02d0540974345534f266b9976a6597e548",
    "dev_path": "data/benchmarks/math_structural_q0_v1.json",
    "holdout_commit": "b7619f80440d12d57bef5e24c3377b158ec5714d",
    "holdout_blob": "ccd0715b1030b88b7ed1d82f52bfbb8ffd70eaac",
    "holdout_path": "crates/core/symthaea-core/examples/support/math_structural_q0_holdout_v1.rs",
    "holdout_id": "math-structural-q0-holdout-v1",
    "qual_commit": "694ef48e53296c2c6fe39a9c07927085b2f2754f",
    "qual_run": 35459503395,
    "qual_workflow": "Math Structural Q0 Qualification",
}

EXPECTED_ENV = {
    "crates/core/symthaea-core/src/hdc/binary_hv.rs": "22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056",
    "crates/core/symthaea-core/src/hdc/fol_formula_ext.rs": "fd1a575a000b214ad56de449ded94458c2f3a3e3",
    "crates/core/symthaea-core/src/hdc/logic_engine.rs": "701f77f53b6aedc27c7498e908611329e892daca",
    "crates/core/symthaea-core/src/hdc/primitive_system/mod.rs": "b65be8cb245f7022752821cb69ad28da0961d931",
    "crates/core/symthaea-core/Cargo.toml": "64c70df9ca1e1ba80e996ef8f2c8ac0d026d45ba",
    "Cargo.lock": "1d5b6c3dffb474fabef0f8237c6741b4f252bc62",
}

FORBIDDEN_AUTHORITY_KEYS = {
    "truth_value",
    "formal_authority",
    "theorem_authority",
    "proof_valid",
    "epistemic_confidence",
    "hdc_advantage",
    "production_ready",
}


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def exact_keys(obj: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(obj)
    require(actual == expected, f"{where}: keys differ; missing={sorted(expected-actual)} extra={sorted(actual-expected)}")


def require_sha40(value: Any, where: str) -> None:
    require(isinstance(value, str) and SHA40.fullmatch(value) is not None, f"{where}: expected 40-hex git object id")


def require_sha256(value: Any, where: str) -> None:
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None, f"{where}: expected 64-hex sha256")


def reject_authority_smuggling(value: Any, where: str = "root") -> None:
    if isinstance(value, dict):
        bad = FORBIDDEN_AUTHORITY_KEYS.intersection(value)
        require(not bad, f"{where}: forbidden authority fields {sorted(bad)}")
        for key, child in value.items():
            reject_authority_smuggling(child, f"{where}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            reject_authority_smuggling(child, f"{where}[{idx}]")


def validate_subject(subject: dict[str, Any]) -> None:
    exact_keys(
        subject,
        {"version", "authority", "preparation_base_commit", "source_lineage", "qualification_gate", "stable_environment_at_preparation", "extraction_policy"},
        "subject",
    )
    require(subject["version"] == EXPECTED["version"], "subject.version mismatch")
    require(subject["authority"] == EXPECTED["authority"], "subject.authority mismatch")
    require(subject["preparation_base_commit"] == EXPECTED["preparation_base_commit"], "subject preparation base mismatch")

    lineage = subject["source_lineage"]
    exact_keys(lineage, {"hdc", "canonical_ast", "development_fixture", "holdout"}, "subject.source_lineage")

    hdc = lineage["hdc"]
    exact_keys(hdc, {"commit", "path", "blob", "encoder_id"}, "subject.source_lineage.hdc")
    require((hdc["commit"], hdc["path"], hdc["blob"], hdc["encoder_id"]) == (EXPECTED["hdc_commit"], EXPECTED["hdc_path"], EXPECTED["hdc_blob"], EXPECTED["hdc_id"]), "frozen HDC subject mismatch")

    ast = lineage["canonical_ast"]
    exact_keys(ast, {"commit", "path", "blob", "encoder_id"}, "subject.source_lineage.canonical_ast")
    require((ast["commit"], ast["path"], ast["blob"], ast["encoder_id"]) == (EXPECTED["ast_commit"], EXPECTED["ast_path"], EXPECTED["ast_blob"], EXPECTED["ast_id"]), "frozen AST subject mismatch")

    dev = lineage["development_fixture"]
    exact_keys(dev, {"commit", "path", "blob"}, "subject.source_lineage.development_fixture")
    require((dev["commit"], dev["path"], dev["blob"]) == (EXPECTED["ast_commit"], EXPECTED["dev_path"], EXPECTED["dev_blob"]), "frozen development fixture mismatch")

    holdout = lineage["holdout"]
    exact_keys(holdout, {"commit", "path", "blob", "holdout_id"}, "subject.source_lineage.holdout")
    require((holdout["commit"], holdout["path"], holdout["blob"], holdout["holdout_id"]) == (EXPECTED["holdout_commit"], EXPECTED["holdout_path"], EXPECTED["holdout_blob"], EXPECTED["holdout_id"]), "frozen holdout mismatch")

    gate = subject["qualification_gate"]
    exact_keys(gate, {"subject_commit", "workflow_name", "workflow_run_id", "required_conclusion"}, "subject.qualification_gate")
    require(gate["subject_commit"] == EXPECTED["qual_commit"], "qualification subject mismatch")
    require(gate["workflow_name"] == EXPECTED["qual_workflow"], "qualification workflow mismatch")
    require(gate["workflow_run_id"] == EXPECTED["qual_run"], "qualification run mismatch")
    require(gate["required_conclusion"] == "success", "qualification must require success")

    env = subject["stable_environment_at_preparation"]
    require(isinstance(env, list) and len(env) == len(EXPECTED_ENV), "stable environment entry count mismatch")
    env_map: dict[str, str] = {}
    for idx, item in enumerate(env):
        exact_keys(item, {"path", "blob"}, f"stable_environment_at_preparation[{idx}]")
        require_sha40(item["blob"], f"stable_environment_at_preparation[{idx}].blob")
        require(item["path"] not in env_map, f"duplicate environment path {item['path']}")
        env_map[item["path"]] = item["blob"]
    require(env_map == EXPECTED_ENV, "stable preparation environment mismatch")

    policy = subject["extraction_policy"]
    exact_keys(policy, {"kind", "representation_rule_changes_allowed", "encoder_ids_may_change", "examples_must_delegate_to_library", "independent_implementation_count_per_representation", "holdout_ranking_evaluation_allowed", "holdout_score_emission_allowed", "blind_holdout_compatibility_digest_allowed"}, "subject.extraction_policy")
    require(policy == {
        "kind": "RelocationOnly",
        "representation_rule_changes_allowed": False,
        "encoder_ids_may_change": False,
        "examples_must_delegate_to_library": True,
        "independent_implementation_count_per_representation": 1,
        "holdout_ranking_evaluation_allowed": False,
        "holdout_score_emission_allowed": False,
        "blind_holdout_compatibility_digest_allowed": True,
    }, "extraction policy mismatch")


def validate_receipt(receipt: dict[str, Any], subject: dict[str, Any]) -> None:
    validate_subject(subject)
    reject_authority_smuggling(receipt)
    exact_keys(receipt, {"version", "authority", "extraction_kind", "target_commit", "qualification", "source_integrity", "target_library", "identity", "delegation", "development_compatibility", "aggregate_compatibility", "holdout_firewall", "evidence_refs"}, "receipt")
    require(receipt["version"] == "math-structural-extraction-receipt-v1", "receipt.version mismatch")
    require(receipt["authority"] == "MeasurementOnly", "receipt.authority must remain MeasurementOnly")
    require(receipt["extraction_kind"] == "RelocationOnly", "extraction_kind must be RelocationOnly")
    require_sha40(receipt["target_commit"], "receipt.target_commit")

    qualification = receipt["qualification"]
    exact_keys(qualification, {"subject_commit", "workflow_run_id", "conclusion"}, "receipt.qualification")
    require(qualification["subject_commit"] == subject["qualification_gate"]["subject_commit"], "receipt qualification subject mismatch")
    require(qualification["workflow_run_id"] == subject["qualification_gate"]["workflow_run_id"], "receipt qualification run mismatch")
    require(qualification["conclusion"] == "success", "predecessor qualification must actually be success")

    integrity = receipt["source_integrity"]
    exact_keys(integrity, {"hdc_source_blob", "ast_source_blob", "development_fixture_blob", "holdout_source_blob"}, "receipt.source_integrity")
    expected_integrity = {
        "hdc_source_blob": subject["source_lineage"]["hdc"]["blob"],
        "ast_source_blob": subject["source_lineage"]["canonical_ast"]["blob"],
        "development_fixture_blob": subject["source_lineage"]["development_fixture"]["blob"],
        "holdout_source_blob": subject["source_lineage"]["holdout"]["blob"],
    }
    require(integrity == expected_integrity, "receipt source integrity does not match frozen subject")

    library = receipt["target_library"]
    exact_keys(library, {"module_path", "hdc_type", "ast_type"}, "receipt.target_library")
    for key, value in library.items():
        require(isinstance(value, str) and value.strip(), f"target_library.{key} must be non-empty")

    identity = receipt["identity"]
    exact_keys(identity, {"hdc_encoder_id", "ast_encoder_id", "representation_rule_changed"}, "receipt.identity")
    require(identity["hdc_encoder_id"] == subject["source_lineage"]["hdc"]["encoder_id"], "HDC encoder identity changed")
    require(identity["ast_encoder_id"] == subject["source_lineage"]["canonical_ast"]["encoder_id"], "AST encoder identity changed")
    require(identity["representation_rule_changed"] is False, "representation-rule changes require a new lineage")

    delegation = receipt["delegation"]
    exact_keys(delegation, {"examples_delegate_to_library", "hdc_implementation_count", "ast_implementation_count"}, "receipt.delegation")
    require(delegation["examples_delegate_to_library"] is True, "examples must delegate to library")
    require(delegation["hdc_implementation_count"] == 1, "exactly one HDC implementation is allowed")
    require(delegation["ast_implementation_count"] == 1, "exactly one AST implementation is allowed")

    cases = receipt["development_compatibility"]
    require(isinstance(cases, list) and cases, "development_compatibility must be non-empty")
    seen: set[str] = set()
    case_keys = {"case_id", "predecessor_hdc_vector_sha256", "extracted_hdc_vector_sha256", "predecessor_sparse_map_sha256", "extracted_sparse_map_sha256", "predecessor_ranking_sha256", "extracted_ranking_sha256"}
    for idx, case in enumerate(cases):
        exact_keys(case, case_keys, f"development_compatibility[{idx}]")
        case_id = case["case_id"]
        require(isinstance(case_id, str) and case_id, f"development_compatibility[{idx}].case_id empty")
        require(case_id not in seen, f"duplicate development case {case_id}")
        seen.add(case_id)
        pairs = [
            ("hdc_vector", case["predecessor_hdc_vector_sha256"], case["extracted_hdc_vector_sha256"]),
            ("sparse_map", case["predecessor_sparse_map_sha256"], case["extracted_sparse_map_sha256"]),
            ("ranking", case["predecessor_ranking_sha256"], case["extracted_ranking_sha256"]),
        ]
        for label, before, after in pairs:
            require_sha256(before, f"{case_id}.{label}.predecessor")
            require_sha256(after, f"{case_id}.{label}.extracted")
            require(before == after, f"{case_id}: {label} changed during extraction")

    aggregate = receipt["aggregate_compatibility"]
    aggregate_keys = {"predecessor_hdc_pairwise_sha256", "extracted_hdc_pairwise_sha256", "predecessor_sparse_cosine_sha256", "extracted_sparse_cosine_sha256", "predecessor_tie_order_sha256", "extracted_tie_order_sha256"}
    exact_keys(aggregate, aggregate_keys, "receipt.aggregate_compatibility")
    for label in ("hdc_pairwise", "sparse_cosine", "tie_order"):
        before = aggregate[f"predecessor_{label}_sha256"]
        after = aggregate[f"extracted_{label}_sha256"]
        require_sha256(before, f"aggregate.{label}.predecessor")
        require_sha256(after, f"aggregate.{label}.extracted")
        require(before == after, f"aggregate {label} changed during extraction")

    holdout = receipt["holdout_firewall"]
    exact_keys(holdout, {"holdout_id", "source_blob", "ranking_evaluated", "scores_emitted", "blind_predecessor_digest", "blind_extracted_digest"}, "receipt.holdout_firewall")
    require(holdout["holdout_id"] == subject["source_lineage"]["holdout"]["holdout_id"], "holdout id mismatch")
    require(holdout["source_blob"] == subject["source_lineage"]["holdout"]["blob"], "holdout source blob mismatch")
    require(holdout["ranking_evaluated"] is False, "holdout ranking must remain unevaluated during extraction")
    require(holdout["scores_emitted"] is False, "holdout scores must not be emitted during extraction")
    require_sha256(holdout["blind_predecessor_digest"], "holdout.blind_predecessor_digest")
    require_sha256(holdout["blind_extracted_digest"], "holdout.blind_extracted_digest")
    require(holdout["blind_predecessor_digest"] == holdout["blind_extracted_digest"], "blind holdout representation digest changed")

    refs = receipt["evidence_refs"]
    require(isinstance(refs, list) and refs, "evidence_refs must be non-empty")
    require(all(isinstance(ref, str) and ref.strip() for ref in refs), "evidence_refs must contain non-empty strings")
    require(len(refs) == len(set(refs)), "evidence_refs must be unique")


def valid_receipt(subject: dict[str, Any]) -> dict[str, Any]:
    d = "ab" * 32
    return {
        "version": "math-structural-extraction-receipt-v1",
        "authority": "MeasurementOnly",
        "extraction_kind": "RelocationOnly",
        "target_commit": "1" * 40,
        "qualification": {
            "subject_commit": subject["qualification_gate"]["subject_commit"],
            "workflow_run_id": subject["qualification_gate"]["workflow_run_id"],
            "conclusion": "success",
        },
        "source_integrity": {
            "hdc_source_blob": subject["source_lineage"]["hdc"]["blob"],
            "ast_source_blob": subject["source_lineage"]["canonical_ast"]["blob"],
            "development_fixture_blob": subject["source_lineage"]["development_fixture"]["blob"],
            "holdout_source_blob": subject["source_lineage"]["holdout"]["blob"],
        },
        "target_library": {
            "module_path": "crates/core/symthaea-core/src/hdc/math_structural_retrieval.rs",
            "hdc_type": "StructuralMathEncoderV1",
            "ast_type": "CanonicalAstEncoderV1",
        },
        "identity": {
            "hdc_encoder_id": "symthaea-math-structural-hdc-v1",
            "ast_encoder_id": "canonical-ast-sparse-v1",
            "representation_rule_changed": False,
        },
        "delegation": {
            "examples_delegate_to_library": True,
            "hdc_implementation_count": 1,
            "ast_implementation_count": 1,
        },
        "development_compatibility": [{
            "case_id": "self-test-case",
            "predecessor_hdc_vector_sha256": d,
            "extracted_hdc_vector_sha256": d,
            "predecessor_sparse_map_sha256": d,
            "extracted_sparse_map_sha256": d,
            "predecessor_ranking_sha256": d,
            "extracted_ranking_sha256": d,
        }],
        "aggregate_compatibility": {
            "predecessor_hdc_pairwise_sha256": d,
            "extracted_hdc_pairwise_sha256": d,
            "predecessor_sparse_cosine_sha256": d,
            "extracted_sparse_cosine_sha256": d,
            "predecessor_tie_order_sha256": d,
            "extracted_tie_order_sha256": d,
        },
        "holdout_firewall": {
            "holdout_id": subject["source_lineage"]["holdout"]["holdout_id"],
            "source_blob": subject["source_lineage"]["holdout"]["blob"],
            "ranking_evaluated": False,
            "scores_emitted": False,
            "blind_predecessor_digest": d,
            "blind_extracted_digest": d,
        },
        "evidence_refs": ["self-test://evidence"],
    }


def expect_reject(subject: dict[str, Any], mutate, label: str) -> None:
    receipt = valid_receipt(subject)
    mutate(receipt)
    try:
        validate_receipt(receipt, subject)
    except ValidationError:
        return
    raise AssertionError(f"negative self-test unexpectedly accepted: {label}")


def run_self_test(subject: dict[str, Any]) -> None:
    validate_subject(subject)
    validate_receipt(valid_receipt(subject), subject)

    expect_reject(subject, lambda r: r.__setitem__("authority", "FormalAuthority"), "authority escalation")
    expect_reject(subject, lambda r: r.__setitem__("extraction_kind", "Rewrite"), "rewrite masquerading as relocation")
    expect_reject(subject, lambda r: r["qualification"].__setitem__("conclusion", "queued"), "unqualified predecessor")
    expect_reject(subject, lambda r: r["source_integrity"].__setitem__("hdc_source_blob", "0" * 40), "source substitution")
    expect_reject(subject, lambda r: r["identity"].__setitem__("hdc_encoder_id", "symthaea-math-structural-hdc-v2"), "encoder identity drift")
    expect_reject(subject, lambda r: r["identity"].__setitem__("representation_rule_changed", True), "representation rewrite")
    expect_reject(subject, lambda r: r["delegation"].__setitem__("hdc_implementation_count", 2), "duplicate HDC implementation")
    expect_reject(subject, lambda r: r["development_compatibility"][0].__setitem__("extracted_hdc_vector_sha256", "cd" * 32), "HDC vector drift")
    expect_reject(subject, lambda r: r["development_compatibility"][0].__setitem__("extracted_sparse_map_sha256", "cd" * 32), "sparse map drift")
    expect_reject(subject, lambda r: r["development_compatibility"][0].__setitem__("extracted_ranking_sha256", "cd" * 32), "ranking drift")
    expect_reject(subject, lambda r: r["aggregate_compatibility"].__setitem__("extracted_tie_order_sha256", "cd" * 32), "tie-order drift")
    expect_reject(subject, lambda r: r["holdout_firewall"].__setitem__("ranking_evaluated", True), "holdout ranking leak")
    expect_reject(subject, lambda r: r["holdout_firewall"].__setitem__("scores_emitted", True), "holdout score leak")
    expect_reject(subject, lambda r: r["holdout_firewall"].__setitem__("blind_extracted_digest", "cd" * 32), "holdout representation drift")
    expect_reject(subject, lambda r: r.__setitem__("truth_value", True), "truth authority smuggling")

    bad_subject = copy.deepcopy(subject)
    bad_subject["source_lineage"]["holdout"]["blob"] = "0" * 40
    try:
        validate_subject(bad_subject)
    except ValidationError:
        pass
    else:
        raise AssertionError("negative self-test unexpectedly accepted: mutated frozen subject")

    print("PASS: math structural extraction receipt semantic self-tests")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    require(isinstance(value, dict), f"{path}: root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        subject = load_json(args.subject)
        validate_subject(subject)
        if args.self_test:
            run_self_test(subject)
        if args.receipt:
            validate_receipt(load_json(args.receipt), subject)
            print(f"PASS: {args.receipt}")
        if not args.self_test and not args.receipt:
            print(f"PASS: {args.subject}")
        return 0
    except (ValidationError, json.JSONDecodeError, OSError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
