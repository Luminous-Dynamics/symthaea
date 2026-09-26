#!/usr/bin/env python3
"""Validate the SYM-FV-006 formal evidence receipt v1 contract.

Zero third-party dependencies. This validates the repository's schema contract and,
when receipt paths are supplied, the semantic invariants JSON Schema alone cannot
express (statement digest, immutable PASS subject, mutation outcome, and evidence
class identity against SYM-FV-000).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs/formal/formal_evidence_receipt_v1.schema.json"
CLASSES_PATH = ROOT / "docs/formal/formal_verification_evidence_classes_v1.json"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SCHEMA = "symthaea.formal-evidence.receipt.v1"
EXPECTED_AUTHORITY = "EvidenceOnly"
EXPECTED_COMPOSITION = "DependenciesDoNotPromotePrimaryEvidenceClass"


class ContractError(ValueError):
    pass


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"{path}: top-level JSON value must be an object")
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_object(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _require_list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an array")
    return value


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be non-empty text")
    return value


def _require_hex(value: object, name: str, regex: re.Pattern[str]) -> str:
    text = _require_text(value, name)
    if regex.fullmatch(text) is None:
        raise ContractError(f"{name} has invalid digest/identity syntax")
    return text


def evidence_classes() -> list[str]:
    contract = _load(CLASSES_PATH)
    classes = contract.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ContractError("evidence-class contract has no classes")
    ids = [entry.get("id") for entry in classes if isinstance(entry, dict)]
    if len(ids) != len(classes) or any(not isinstance(item, str) for item in ids):
        raise ContractError("evidence-class contract contains malformed entries")
    if len(ids) != len(set(ids)):
        raise ContractError("evidence-class IDs must be unique")
    return ids


def validate_schema_contract(schema: dict, classes: list[str]) -> None:
    required = schema.get("required")
    if not isinstance(required, list):
        raise ContractError("schema.required must be an array")

    must_require = {
        "schema",
        "authority",
        "primary_evidence_class",
        "subject",
        "claim",
        "toolchain",
        "artifacts",
        "trust_boundary",
        "controls",
        "qualification",
        "immutability",
        "composition",
    }
    missing = sorted(must_require - set(required))
    if missing:
        raise ContractError(f"schema is missing required fields: {missing}")

    props = _require_object(schema.get("properties"), "schema.properties")
    if _require_object(props.get("schema"), "schema.properties.schema").get("const") != EXPECTED_SCHEMA:
        raise ContractError("schema identity drift")
    if _require_object(props.get("authority"), "schema.properties.authority").get("const") != EXPECTED_AUTHORITY:
        raise ContractError("authority must remain EvidenceOnly")

    primary = _require_object(
        props.get("primary_evidence_class"), "schema.properties.primary_evidence_class"
    ).get("enum")
    if primary != classes:
        raise ContractError(
            "receipt primary_evidence_class enum must exactly match SYM-FV-000 evidence classes"
        )

    composition = _require_object(props.get("composition"), "schema.properties.composition")
    comp_props = _require_object(composition.get("properties"), "composition.properties")
    rule = _require_object(comp_props.get("composition_rule"), "composition_rule")
    if rule.get("const") != EXPECTED_COMPOSITION:
        raise ContractError("composition must not promote the primary evidence class")

    dep = _require_object(comp_props.get("dependencies"), "composition.dependencies")
    dep_item = _require_object(dep.get("items"), "composition.dependencies.items")
    dep_props = _require_object(dep_item.get("properties"), "dependency.properties")
    dep_classes = _require_object(dep_props.get("evidence_class"), "dependency.evidence_class").get("enum")
    if dep_classes != classes:
        raise ContractError("dependency evidence-class enum drift")

    if schema.get("additionalProperties") is not False:
        raise ContractError("receipt schema must fail closed on unknown top-level fields")


def validate_receipt(receipt: dict, classes: list[str]) -> None:
    if receipt.get("schema") != EXPECTED_SCHEMA:
        raise ContractError("receipt schema identity mismatch")
    if receipt.get("authority") != EXPECTED_AUTHORITY:
        raise ContractError("receipt authority must be EvidenceOnly")

    primary = receipt.get("primary_evidence_class")
    if not isinstance(primary, str) or primary not in classes:
        raise ContractError("receipt must name exactly one admitted primary evidence class")

    subject = _require_object(receipt.get("subject"), "subject")
    _require_text(subject.get("repository"), "subject.repository")
    _require_hex(subject.get("commit"), "subject.commit", HEX40)
    _require_text(subject.get("path"), "subject.path")
    _require_hex(subject.get("blob"), "subject.blob", HEX40)
    _require_text(subject.get("symbol"), "subject.symbol")
    if "source_sha256" in subject:
        _require_hex(subject.get("source_sha256"), "subject.source_sha256", HEX64)

    claim = _require_object(receipt.get("claim"), "claim")
    _require_text(claim.get("claim_id"), "claim.claim_id")
    statement = _require_text(claim.get("statement"), "claim.statement")
    statement_digest = _require_hex(
        claim.get("statement_sha256"), "claim.statement_sha256", HEX64
    )
    if _sha256_text(statement) != statement_digest:
        raise ContractError("claim.statement_sha256 does not match claim.statement bytes")
    if claim.get("elaborated_statement_sha256") is not None:
        _require_hex(
            claim.get("elaborated_statement_sha256"),
            "claim.elaborated_statement_sha256",
            HEX64,
        )
    _require_text(claim.get("claim_ceiling"), "claim.claim_ceiling")
    nonclaims = _require_list(claim.get("nonclaims"), "claim.nonclaims")
    if not nonclaims or any(not isinstance(item, str) or not item for item in nonclaims):
        raise ContractError("claim.nonclaims must contain non-empty strings")

    toolchain = _require_list(receipt.get("toolchain"), "toolchain")
    if not toolchain:
        raise ContractError("toolchain must not be empty")
    names = []
    for idx, entry in enumerate(toolchain):
        item = _require_object(entry, f"toolchain[{idx}]")
        names.append(_require_text(item.get("name"), f"toolchain[{idx}].name"))
        _require_text(item.get("identity"), f"toolchain[{idx}].identity")
        _require_text(item.get("role"), f"toolchain[{idx}].role")
    if len(names) != len(set(names)):
        raise ContractError("toolchain names must be unique")

    for idx, artifact in enumerate(_require_list(receipt.get("artifacts"), "artifacts")):
        item = _require_object(artifact, f"artifacts[{idx}]")
        _require_text(item.get("kind"), f"artifacts[{idx}].kind")
        _require_text(item.get("path_or_name"), f"artifacts[{idx}].path_or_name")
        _require_hex(item.get("sha256"), f"artifacts[{idx}].sha256", HEX64)

    trust = _require_object(receipt.get("trust_boundary"), "trust_boundary")
    _require_list(trust.get("assumptions"), "trust_boundary.assumptions")
    external = _require_list(trust.get("external_models"), "trust_boundary.external_models")
    _require_list(
        trust.get("axiom_or_assumption_census"),
        "trust_boundary.axiom_or_assumption_census",
    )
    for idx, model in enumerate(external):
        item = _require_object(model, f"external_models[{idx}]")
        _require_text(item.get("identity"), f"external_models[{idx}].identity")
        _require_hex(item.get("sha256"), f"external_models[{idx}].sha256", HEX64)
        if item.get("status") not in {"Proved", "Trusted", "Unmodeled", "UnsupportedBoundary"}:
            raise ContractError(f"external_models[{idx}].status is invalid")

    controls = _require_object(receipt.get("controls"), "controls")
    mutations = _require_list(controls.get("mutations"), "controls.mutations")
    for idx, mutation in enumerate(mutations):
        item = _require_object(mutation, f"mutations[{idx}]")
        _require_text(item.get("id"), f"mutations[{idx}].id")
        _require_text(item.get("expected_detection"), f"mutations[{idx}].expected_detection")
        if item.get("observed_result") not in {
            "ExpectedFail",
            "UnexpectedPass",
            "NotRun",
            "InfrastructureFailure",
        }:
            raise ContractError(f"mutations[{idx}].observed_result is invalid")

    qualification = _require_object(receipt.get("qualification"), "qualification")
    result = qualification.get("result")
    if result not in {"Pass", "Fail", "Blocked", "EnvironmentFailure"}:
        raise ContractError("qualification.result is invalid")
    _require_text(qualification.get("workflow"), "qualification.workflow")
    if not isinstance(qualification.get("run_id"), (str, int)):
        raise ContractError("qualification.run_id must be text or integer")
    _require_hex(qualification.get("head_sha"), "qualification.head_sha", HEX40)

    immutability = _require_object(receipt.get("immutability"), "immutability")
    pre = _require_hex(
        immutability.get("pre_subject_sha256"), "immutability.pre_subject_sha256", HEX64
    )
    post = _require_hex(
        immutability.get("post_subject_sha256"), "immutability.post_subject_sha256", HEX64
    )
    clean = immutability.get("checkout_clean")
    if not isinstance(clean, bool):
        raise ContractError("immutability.checkout_clean must be boolean")

    composition = _require_object(receipt.get("composition"), "composition")
    if composition.get("composition_rule") != EXPECTED_COMPOSITION:
        raise ContractError("composition rule drift")
    deps = _require_list(composition.get("dependencies"), "composition.dependencies")
    dep_digests = []
    for idx, dep in enumerate(deps):
        item = _require_object(dep, f"dependencies[{idx}]")
        dep_digests.append(
            _require_hex(item.get("receipt_sha256"), f"dependencies[{idx}].receipt_sha256", HEX64)
        )
        if item.get("evidence_class") not in classes:
            raise ContractError(f"dependencies[{idx}].evidence_class is invalid")
        if item.get("relation") not in {
            "depends_on",
            "refines",
            "conforms_to",
            "qualified_by",
            "imports_assurance_from",
        }:
            raise ContractError(f"dependencies[{idx}].relation is invalid")
    if len(dep_digests) != len(set(dep_digests)):
        raise ContractError("composition dependencies must not duplicate receipt digests")

    if result == "Pass":
        if pre != post or not clean:
            raise ContractError("PASS requires immutable subject bytes and a clean checkout")
        bad_mutants = [
            m for m in mutations if m.get("observed_result") != "ExpectedFail"
        ]
        if bad_mutants:
            raise ContractError("PASS requires every declared mutation control to ExpectedFail")
        if primary in {"ExtractedSourceRefinement", "DeductiveImplementationProof"}:
            if any(model.get("status") == "UnsupportedBoundary" for model in external):
                raise ContractError(
                    "deductive/refinement PASS cannot retain an UnsupportedBoundary external model"
                )


def _fixture(classes: list[str]) -> dict:
    statement = "forall a b, extracted_bind(a,b) = abstract_bind(a,b)"
    z64 = "0" * 64
    return {
        "schema": EXPECTED_SCHEMA,
        "tracking_issue": 5719,
        "authority": EXPECTED_AUTHORITY,
        "primary_evidence_class": "ExtractedSourceRefinement",
        "subject": {
            "repository": "Luminous-Dynamics/symthaea",
            "commit": "1" * 40,
            "path": "crates/core/example.rs",
            "blob": "2" * 40,
            "symbol": "example::bind",
            "source_sha256": "3" * 64,
        },
        "claim": {
            "claim_id": "SYM-FV-006-SELFTEST",
            "statement": statement,
            "statement_sha256": _sha256_text(statement),
            "elaborated_statement_sha256": None,
            "claim_ceiling": "self-test only",
            "nonclaims": ["runtime authority"],
        },
        "toolchain": [
            {"name": "Lean4", "identity": "self-test", "role": "kernel", "options": []}
        ],
        "artifacts": [{"kind": "proof", "path_or_name": "example", "sha256": "4" * 64}],
        "trust_boundary": {
            "assumptions": [],
            "external_models": [],
            "axiom_or_assumption_census": [],
        },
        "controls": {
            "mutations": [
                {
                    "id": "wrong-operator",
                    "expected_detection": "semantic theorem mismatch",
                    "observed_result": "ExpectedFail",
                }
            ]
        },
        "qualification": {
            "result": "Pass",
            "workflow": "self-test",
            "run_id": "self-test",
            "head_sha": "1" * 40,
            "notes": "synthetic validator self-test; never retained as evidence",
        },
        "immutability": {
            "pre_subject_sha256": z64,
            "post_subject_sha256": z64,
            "checkout_clean": True,
        },
        "composition": {
            "dependencies": [],
            "composition_rule": EXPECTED_COMPOSITION,
        },
    }


def self_test(classes: list[str]) -> None:
    valid = _fixture(classes)
    validate_receipt(valid, classes)

    mutant = json.loads(json.dumps(valid))
    mutant["claim"]["statement"] += " "
    try:
        validate_receipt(mutant, classes)
    except ContractError:
        pass
    else:
        raise ContractError("self-test failed: statement-digest drift was accepted")

    mutant = json.loads(json.dumps(valid))
    mutant["immutability"]["post_subject_sha256"] = "f" * 64
    try:
        validate_receipt(mutant, classes)
    except ContractError:
        pass
    else:
        raise ContractError("self-test failed: mutable PASS subject was accepted")

    mutant = json.loads(json.dumps(valid))
    mutant["controls"]["mutations"][0]["observed_result"] = "UnexpectedPass"
    try:
        validate_receipt(mutant, classes)
    except ContractError:
        pass
    else:
        raise ContractError("self-test failed: unexpected-pass mutant was accepted")

    mutant = json.loads(json.dumps(valid))
    mutant["primary_evidence_class"] = ["AbstractFormalTheorem", "RuntimeQualification"]
    try:
        validate_receipt(mutant, classes)
    except ContractError:
        pass
    else:
        raise ContractError("self-test failed: multiple primary evidence classes were accepted")


def main(argv: list[str]) -> int:
    classes = evidence_classes()
    schema = _load(SCHEMA_PATH)
    validate_schema_contract(schema, classes)
    self_test(classes)

    for raw in argv[1:]:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        validate_receipt(_load(path), classes)

    print("FORMAL_EVIDENCE_RECEIPT_V1_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except ContractError as exc:
        print(f"FORMAL_EVIDENCE_RECEIPT_V1_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
