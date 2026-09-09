#!/usr/bin/env python3
"""Compose an immutable historical trusted-main root receipt.

This tool composes already-produced protection evidence into a content-addressed
historical root record. It is deliberately non-scientific and non-attesting:
it does not apply GitHub settings, authenticate operators, establish current
admission, activate a self-hosted runner, or retroactively authorize bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "symthaea.github-trusted-main-protection-policy.v1"
STRUCTURAL_SCHEMA = "symthaea.github-trusted-main-protection-verification.v1"
EFFECTIVE_SCHEMA = "symthaea.github-trusted-main-effective-rules-verification.v1"
ROOT_SUBJECT_SCHEMA = "symthaea.github-trusted-main-root-subject.v1"
ENFORCEMENT_SCHEMA = "symthaea.github-trusted-main-enforcement-evidence.v1"
ROOT_RECEIPT_SCHEMA = "symthaea.github-trusted-main-root-receipt.v1"
BYPASS_POLICY_SCHEMA = "symthaea.github-trusted-main-bypass-policy.v1"

POLICY_DOMAIN = b"symthaea.github-trusted-main-protection-policy.v1\0"
STRUCTURAL_DOMAIN = b"symthaea.github-trusted-main-protection-verification.v1\0"
EFFECTIVE_DOMAIN = b"symthaea.github-trusted-main-effective-rules-verification.v1\0"
ROOT_SUBJECT_DOMAIN = b"symthaea.github-trusted-main-root-subject.v1\0"
ENFORCEMENT_DOMAIN = b"symthaea.github-trusted-main-enforcement-evidence.v1\0"
ROOT_RECEIPT_DOMAIN = b"symthaea.github-trusted-main-root-receipt.v1\0"
BYPASS_POLICY_DOMAIN = b"symthaea.github-trusted-main-bypass-policy.v1\0"

MAX_JSON_BYTES = 1_000_000
SUPPORTED_RULES = frozenset({"deletion", "non_fast_forward", "pull_request"})
MERGE_METHODS = frozenset({"merge", "rebase", "squash"})
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RFC3339_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class RootReceiptError(ValueError):
    """Fail-closed root receipt validation/composition error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RootReceiptError(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise RootReceiptError(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except RootReceiptError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RootReceiptError(f"{path}: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + _canonical(value)).hexdigest()


def _exact_keys(value: dict[str, Any], required: set[str], *, where: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise RootReceiptError(f"{where}: missing fields: {', '.join(missing)}")
    if unknown:
        raise RootReceiptError(f"{where}: unknown fields: {', '.join(unknown)}")


def _string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RootReceiptError(f"{where}: non-empty string required")
    if value != value.strip():
        raise RootReceiptError(f"{where}: surrounding whitespace is non-canonical")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise RootReceiptError(f"{where}: control characters are forbidden")
    return value


def _int(value: Any, *, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RootReceiptError(f"{where}: integer >= {minimum} required")
    return value


def _bool(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise RootReceiptError(f"{where}: boolean required")
    return value


def _sha40(value: Any, *, where: str) -> str:
    value = _string(value, where=where)
    if SHA40_RE.fullmatch(value) is None:
        raise RootReceiptError(f"{where}: expected 40 lowercase hex")
    return value


def _sha256_id(value: Any, *, where: str) -> str:
    value = _string(value, where=where)
    if SHA256_ID_RE.fullmatch(value) is None:
        raise RootReceiptError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _sorted_unique_strings(value: Any, *, where: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise RootReceiptError(f"{where}: array required")
    items = [_string(item, where=f"{where}[]") for item in value]
    if items != sorted(set(items)):
        raise RootReceiptError(f"{where}: must be sorted and unique")
    if not allow_empty and not items:
        raise RootReceiptError(f"{where}: must not be empty")
    return items


def normalize_policy(raw: Any) -> dict[str, Any]:
    """Mirror the P0 policy normalization so PolicyId is independently derived."""
    if not isinstance(raw, dict):
        raise RootReceiptError("policy: object required")
    _exact_keys(raw, {
        "schema", "repository", "repository_id", "target_ref", "ruleset_name",
        "required_enforcement", "allowed_bypass_actors", "required_rules",
        "pull_request_policy", "p0_required_status_checks", "non_claims",
    }, where="policy")
    if raw["schema"] != POLICY_SCHEMA:
        raise RootReceiptError(f"policy.schema: expected {POLICY_SCHEMA!r}")

    repository = _string(raw["repository"], where="policy.repository")
    repository_id = _int(raw["repository_id"], where="policy.repository_id", minimum=1)
    target_ref = _string(raw["target_ref"], where="policy.target_ref")
    if not target_ref.startswith("refs/heads/"):
        raise RootReceiptError("policy.target_ref: full refs/heads/... ref required")
    ruleset_name = _string(raw["ruleset_name"], where="policy.ruleset_name")
    if raw["required_enforcement"] != "active":
        raise RootReceiptError("policy.required_enforcement: P0 requires active")

    bypass = raw["allowed_bypass_actors"]
    if not isinstance(bypass, list):
        raise RootReceiptError("policy.allowed_bypass_actors: array required")
    normalized_bypass: list[dict[str, Any]] = []
    for index, actor in enumerate(bypass):
        if not isinstance(actor, dict):
            raise RootReceiptError(f"policy.allowed_bypass_actors[{index}]: object required")
        _exact_keys(actor, {"actor_id", "actor_type", "bypass_mode"}, where=f"policy.allowed_bypass_actors[{index}]")
        normalized_bypass.append({
            "actor_id": _int(actor["actor_id"], where=f"policy.allowed_bypass_actors[{index}].actor_id", minimum=1),
            "actor_type": _string(actor["actor_type"], where=f"policy.allowed_bypass_actors[{index}].actor_type"),
            "bypass_mode": _string(actor["bypass_mode"], where=f"policy.allowed_bypass_actors[{index}].bypass_mode"),
        })
    bypass_keys = [(a["actor_type"], a["actor_id"], a["bypass_mode"]) for a in normalized_bypass]
    if bypass_keys != sorted(set(bypass_keys)):
        raise RootReceiptError("policy.allowed_bypass_actors: must be sorted and unique")

    required_rules = _sorted_unique_strings(raw["required_rules"], where="policy.required_rules", allow_empty=False)
    if set(required_rules) != SUPPORTED_RULES:
        raise RootReceiptError("policy.required_rules: P0 requires exactly deletion, non_fast_forward, pull_request")

    pr = raw["pull_request_policy"]
    if not isinstance(pr, dict):
        raise RootReceiptError("policy.pull_request_policy: object required")
    _exact_keys(pr, {
        "required_approving_review_count_min", "required_review_thread_resolution",
        "require_code_owner_review", "require_last_push_approval",
        "dismiss_stale_reviews_on_push", "allowed_merge_methods",
    }, where="policy.pull_request_policy")
    methods = _sorted_unique_strings(pr["allowed_merge_methods"], where="policy.pull_request_policy.allowed_merge_methods", allow_empty=False)
    if not set(methods) <= MERGE_METHODS:
        raise RootReceiptError("policy.pull_request_policy.allowed_merge_methods: unsupported method")
    normalized_pr = {
        "required_approving_review_count_min": _int(pr["required_approving_review_count_min"], where="policy.pull_request_policy.required_approving_review_count_min"),
        "required_review_thread_resolution": _bool(pr["required_review_thread_resolution"], where="policy.pull_request_policy.required_review_thread_resolution"),
        "require_code_owner_review": _bool(pr["require_code_owner_review"], where="policy.pull_request_policy.require_code_owner_review"),
        "require_last_push_approval": _bool(pr["require_last_push_approval"], where="policy.pull_request_policy.require_last_push_approval"),
        "dismiss_stale_reviews_on_push": _bool(pr["dismiss_stale_reviews_on_push"], where="policy.pull_request_policy.dismiss_stale_reviews_on_push"),
        "allowed_merge_methods": methods,
    }
    if raw["p0_required_status_checks"] != []:
        raise RootReceiptError("policy.p0_required_status_checks: P0 intentionally requires none")

    return {
        "schema": POLICY_SCHEMA,
        "repository": repository,
        "repository_id": repository_id,
        "target_ref": target_ref,
        "ruleset_name": ruleset_name,
        "required_enforcement": "active",
        "allowed_bypass_actors": normalized_bypass,
        "required_rules": required_rules,
        "pull_request_policy": normalized_pr,
        "p0_required_status_checks": [],
        "non_claims": _sorted_unique_strings(raw["non_claims"], where="policy.non_claims", allow_empty=False),
    }


def policy_id(policy: Any) -> str:
    return _content_id(POLICY_DOMAIN, normalize_policy(policy))


def bypass_policy_id(policy: Any) -> str:
    p = normalize_policy(policy)
    return _content_id(BYPASS_POLICY_DOMAIN, {
        "schema": BYPASS_POLICY_SCHEMA,
        "repository_id": p["repository_id"],
        "target_ref": p["target_ref"],
        "allowed_bypass_actors": p["allowed_bypass_actors"],
    })


def _validate_derived_id(raw: dict[str, Any], *, domain: bytes, id_field: str, where: str) -> None:
    observed = _sha256_id(raw.get(id_field), where=f"{where}.{id_field}")
    payload = dict(raw)
    payload.pop(id_field)
    expected = _content_id(domain, payload)
    if observed != expected:
        raise RootReceiptError(f"{where}.{id_field}: content ID mismatch")


def normalize_structural(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RootReceiptError("structural verification: object required")
    _exact_keys(raw, {
        "schema", "policy_id", "repository", "repository_id", "target_ref",
        "ruleset_id", "ruleset_source_type", "ruleset_source", "disposition",
        "violations", "enforcement_claim", "negative_push_test",
        "organization_rules", "scientific_authority", "verification_id",
    }, where="structural")
    if raw["schema"] != STRUCTURAL_SCHEMA:
        raise RootReceiptError(f"structural.schema: expected {STRUCTURAL_SCHEMA!r}")
    out = {
        "schema": STRUCTURAL_SCHEMA,
        "policy_id": _sha256_id(raw["policy_id"], where="structural.policy_id"),
        "repository": _string(raw["repository"], where="structural.repository"),
        "repository_id": _int(raw["repository_id"], where="structural.repository_id", minimum=1),
        "target_ref": _string(raw["target_ref"], where="structural.target_ref"),
        "ruleset_id": _int(raw["ruleset_id"], where="structural.ruleset_id", minimum=1),
        "ruleset_source_type": _string(raw["ruleset_source_type"], where="structural.ruleset_source_type"),
        "ruleset_source": _string(raw["ruleset_source"], where="structural.ruleset_source"),
        "disposition": _string(raw["disposition"], where="structural.disposition"),
        "violations": _sorted_unique_strings(raw["violations"], where="structural.violations"),
        "enforcement_claim": _string(raw["enforcement_claim"], where="structural.enforcement_claim"),
        "negative_push_test": _string(raw["negative_push_test"], where="structural.negative_push_test"),
        "organization_rules": _string(raw["organization_rules"], where="structural.organization_rules"),
        "scientific_authority": _string(raw["scientific_authority"], where="structural.scientific_authority"),
        "verification_id": _sha256_id(raw["verification_id"], where="structural.verification_id"),
    }
    _validate_derived_id(out, domain=STRUCTURAL_DOMAIN, id_field="verification_id", where="structural")
    if out["scientific_authority"] != "none":
        raise RootReceiptError("structural.scientific_authority: expected none")
    return out


def normalize_effective(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RootReceiptError("effective verification: object required")
    _exact_keys(raw, {
        "schema", "policy_id", "repository", "repository_id", "target_ref",
        "ruleset_id", "observed_p0_rule_types", "disposition", "violations",
        "enforcement_claim", "negative_push_test", "scientific_authority",
        "verification_id",
    }, where="effective")
    if raw["schema"] != EFFECTIVE_SCHEMA:
        raise RootReceiptError(f"effective.schema: expected {EFFECTIVE_SCHEMA!r}")
    out = {
        "schema": EFFECTIVE_SCHEMA,
        "policy_id": _sha256_id(raw["policy_id"], where="effective.policy_id"),
        "repository": _string(raw["repository"], where="effective.repository"),
        "repository_id": _int(raw["repository_id"], where="effective.repository_id", minimum=1),
        "target_ref": _string(raw["target_ref"], where="effective.target_ref"),
        "ruleset_id": _int(raw["ruleset_id"], where="effective.ruleset_id", minimum=1),
        "observed_p0_rule_types": _sorted_unique_strings(raw["observed_p0_rule_types"], where="effective.observed_p0_rule_types"),
        "disposition": _string(raw["disposition"], where="effective.disposition"),
        "violations": _sorted_unique_strings(raw["violations"], where="effective.violations"),
        "enforcement_claim": _string(raw["enforcement_claim"], where="effective.enforcement_claim"),
        "negative_push_test": _string(raw["negative_push_test"], where="effective.negative_push_test"),
        "scientific_authority": _string(raw["scientific_authority"], where="effective.scientific_authority"),
        "verification_id": _sha256_id(raw["verification_id"], where="effective.verification_id"),
    }
    _validate_derived_id(out, domain=EFFECTIVE_DOMAIN, id_field="verification_id", where="effective")
    if out["scientific_authority"] != "none":
        raise RootReceiptError("effective.scientific_authority: expected none")
    return out


def normalize_root_subject(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RootReceiptError("root subject: object required")
    _exact_keys(raw, {
        "schema", "repository", "repository_id", "target_ref", "root_sha",
        "root_tree", "observation_basis",
    }, where="root_subject")
    if raw["schema"] != ROOT_SUBJECT_SCHEMA:
        raise RootReceiptError(f"root_subject.schema: expected {ROOT_SUBJECT_SCHEMA!r}")
    out = {
        "schema": ROOT_SUBJECT_SCHEMA,
        "repository": _string(raw["repository"], where="root_subject.repository"),
        "repository_id": _int(raw["repository_id"], where="root_subject.repository_id", minimum=1),
        "target_ref": _string(raw["target_ref"], where="root_subject.target_ref"),
        "root_sha": _sha40(raw["root_sha"], where="root_subject.root_sha"),
        "root_tree": _sha40(raw["root_tree"], where="root_subject.root_tree"),
        "observation_basis": _string(raw["observation_basis"], where="root_subject.observation_basis"),
    }
    if out["observation_basis"] != "github-commit-readback":
        raise RootReceiptError("root_subject.observation_basis: expected github-commit-readback")
    return out


def root_subject_id(root_subject: Any) -> str:
    return _content_id(ROOT_SUBJECT_DOMAIN, normalize_root_subject(root_subject))


def normalize_enforcement(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RootReceiptError("enforcement evidence: object required")
    _exact_keys(raw, {
        "schema", "repository", "repository_id", "target_ref", "ruleset_id",
        "policy_id", "structural_verification_id", "effective_rules_verification_id",
        "root_sha", "method", "disposition", "ordinary_direct_push",
        "force_push", "deletion", "bypass_assurance", "actor_or_scope",
        "rule_suite_ids", "recorded_at", "scientific_authority", "evidence_id",
    }, where="enforcement")
    if raw["schema"] != ENFORCEMENT_SCHEMA:
        raise RootReceiptError(f"enforcement.schema: expected {ENFORCEMENT_SCHEMA!r}")
    ids = raw["rule_suite_ids"]
    if not isinstance(ids, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in ids):
        raise RootReceiptError("enforcement.rule_suite_ids: positive-integer array required")
    if ids != sorted(set(ids)):
        raise RootReceiptError("enforcement.rule_suite_ids: must be sorted and unique")
    out = {
        "schema": ENFORCEMENT_SCHEMA,
        "repository": _string(raw["repository"], where="enforcement.repository"),
        "repository_id": _int(raw["repository_id"], where="enforcement.repository_id", minimum=1),
        "target_ref": _string(raw["target_ref"], where="enforcement.target_ref"),
        "ruleset_id": _int(raw["ruleset_id"], where="enforcement.ruleset_id", minimum=1),
        "policy_id": _sha256_id(raw["policy_id"], where="enforcement.policy_id"),
        "structural_verification_id": _sha256_id(raw["structural_verification_id"], where="enforcement.structural_verification_id"),
        "effective_rules_verification_id": _sha256_id(raw["effective_rules_verification_id"], where="enforcement.effective_rules_verification_id"),
        "root_sha": _sha40(raw["root_sha"], where="enforcement.root_sha"),
        "method": _string(raw["method"], where="enforcement.method"),
        "disposition": _string(raw["disposition"], where="enforcement.disposition"),
        "ordinary_direct_push": _string(raw["ordinary_direct_push"], where="enforcement.ordinary_direct_push"),
        "force_push": _string(raw["force_push"], where="enforcement.force_push"),
        "deletion": _string(raw["deletion"], where="enforcement.deletion"),
        "bypass_assurance": _string(raw["bypass_assurance"], where="enforcement.bypass_assurance"),
        "actor_or_scope": _string(raw["actor_or_scope"], where="enforcement.actor_or_scope"),
        "rule_suite_ids": ids,
        "recorded_at": _string(raw["recorded_at"], where="enforcement.recorded_at"),
        "scientific_authority": _string(raw["scientific_authority"], where="enforcement.scientific_authority"),
        "evidence_id": _sha256_id(raw["evidence_id"], where="enforcement.evidence_id"),
    }
    if RFC3339_Z_RE.fullmatch(out["recorded_at"]) is None:
        raise RootReceiptError("enforcement.recorded_at: canonical UTC YYYY-MM-DDTHH:MM:SSZ required")
    if out["method"] not in {"administrative-verification", "non-bypass-negative-ref-update"}:
        raise RootReceiptError("enforcement.method: unsupported method")
    if out["disposition"] not in {"EnforcementSatisfied", "EnforcementInconclusive"}:
        raise RootReceiptError("enforcement.disposition: unsupported disposition")
    if out["scientific_authority"] != "none":
        raise RootReceiptError("enforcement.scientific_authority: expected none")
    if out["disposition"] == "EnforcementSatisfied":
        expected = {
            "ordinary_direct_push": "blocked",
            "force_push": "blocked",
            "deletion": "blocked",
            "bypass_assurance": "policy-matched",
        }
        for field, required in expected.items():
            if out[field] != required:
                raise RootReceiptError(f"enforcement.{field}: EnforcementSatisfied requires {required!r}")
    _validate_derived_id(out, domain=ENFORCEMENT_DOMAIN, id_field="evidence_id", where="enforcement")
    return out


def enforcement_evidence_id(evidence_without_id: dict[str, Any]) -> str:
    if "evidence_id" in evidence_without_id:
        raise RootReceiptError("enforcement_evidence_id: input must not include evidence_id")
    return _content_id(ENFORCEMENT_DOMAIN, evidence_without_id)


def _review_assurance(policy: dict[str, Any]) -> str:
    count = policy["pull_request_policy"]["required_approving_review_count_min"]
    return "pr-mediated-required-approval" if count >= 1 else "pr-mediated-no-required-approval"


def _validate_cross_bindings(
    policy: dict[str, Any],
    structural: dict[str, Any],
    effective: dict[str, Any],
    root: dict[str, Any],
    enforcement: dict[str, Any] | None,
) -> None:
    pid = policy_id(policy)
    expected = (policy["repository"], policy["repository_id"], policy["target_ref"])
    for name, item in (("structural", structural), ("effective", effective), ("root_subject", root)):
        observed = (item["repository"], item["repository_id"], item["target_ref"])
        if observed != expected:
            raise RootReceiptError(f"{name}: repository/ref identity mismatch")
    if structural["policy_id"] != pid or effective["policy_id"] != pid:
        raise RootReceiptError("protection verification policy ID mismatch")
    if structural["ruleset_id"] != effective["ruleset_id"]:
        raise RootReceiptError("protection verification ruleset ID mismatch")
    if structural["ruleset_source_type"] != "Repository" or structural["ruleset_source"] != policy["repository"]:
        raise RootReceiptError("structural verification is not repository-owned")
    if enforcement is not None:
        observed = (enforcement["repository"], enforcement["repository_id"], enforcement["target_ref"])
        if observed != expected:
            raise RootReceiptError("enforcement: repository/ref identity mismatch")
        if enforcement["policy_id"] != pid:
            raise RootReceiptError("enforcement: policy ID mismatch")
        if enforcement["ruleset_id"] != structural["ruleset_id"]:
            raise RootReceiptError("enforcement: ruleset ID mismatch")
        if enforcement["structural_verification_id"] != structural["verification_id"]:
            raise RootReceiptError("enforcement: structural verification ID mismatch")
        if enforcement["effective_rules_verification_id"] != effective["verification_id"]:
            raise RootReceiptError("enforcement: effective-rules verification ID mismatch")
        if enforcement["root_sha"] != root["root_sha"]:
            raise RootReceiptError("enforcement: root SHA mismatch")


def compose_root_receipt(
    *,
    policy: Any,
    structural_verification: Any,
    effective_rules_verification: Any,
    root_subject: Any,
    enforcement_evidence: Any | None = None,
    root_harness_identity: str = "none-pre-bootstrap",
) -> dict[str, Any]:
    p = normalize_policy(policy)
    structural = normalize_structural(structural_verification)
    effective = normalize_effective(effective_rules_verification)
    root = normalize_root_subject(root_subject)
    enforcement = normalize_enforcement(enforcement_evidence) if enforcement_evidence is not None else None
    _validate_cross_bindings(p, structural, effective, root, enforcement)

    if root_harness_identity != "none-pre-bootstrap":
        _sha256_id(root_harness_identity, where="root_harness_identity")

    reasons: list[str] = []
    if structural["disposition"] != "P0StructurallySatisfied" or structural["violations"]:
        reasons.append("StructuralProtectionNotSatisfied")
    if effective["disposition"] != "P0EffectiveRulesSatisfied" or effective["violations"]:
        reasons.append("EffectiveRulesNotSatisfied")
    if enforcement is None:
        reasons.append("EnforcementEvidenceMissing")
    elif enforcement["disposition"] != "EnforcementSatisfied":
        reasons.append("EnforcementEvidenceInconclusive")

    if not reasons:
        disposition = "AdmittedHistoricalRoot"
    elif reasons == ["EnforcementEvidenceMissing"]:
        disposition = "PendingEnforcementEvidence"
    elif reasons == ["EnforcementEvidenceInconclusive"]:
        disposition = "EnforcementInconclusive"
    else:
        disposition = "ProtectionEvidenceNotSatisfied"

    receipt: dict[str, Any] = {
        "schema": ROOT_RECEIPT_SCHEMA,
        "repository": p["repository"],
        "repository_id": p["repository_id"],
        "target_ref": p["target_ref"],
        "root_sha": root["root_sha"],
        "root_tree": root["root_tree"],
        "root_subject_id": root_subject_id(root),
        "protection_policy_id": policy_id(p),
        "protection_verification_id": structural["verification_id"],
        "effective_rules_verification_id": effective["verification_id"],
        "enforcement_evidence_id": enforcement["evidence_id"] if enforcement is not None else None,
        "ruleset_id": structural["ruleset_id"],
        "bypass_policy_id": bypass_policy_id(p),
        "review_assurance": _review_assurance(p),
        "root_harness_identity": root_harness_identity,
        "admission_disposition": disposition,
        "non_admission_reasons": sorted(reasons),
        "historical_scope": "exact-root-only",
        "current_admission": "not-evaluated",
        "receipt_attestation": "none",
        "self_hosted_runner_activation": "not-authorized",
        "scientific_authority": "none",
        "bootstrap_authority": "none",
    }
    receipt["root_receipt_id"] = _content_id(ROOT_RECEIPT_DOMAIN, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("structural_verification", type=Path)
    parser.add_argument("effective_rules_verification", type=Path)
    parser.add_argument("root_subject", type=Path)
    parser.add_argument("--enforcement-evidence", type=Path)
    parser.add_argument("--root-harness-identity", default="none-pre-bootstrap")
    parser.add_argument("--conformance", action="store_true", help="exit zero for valid non-admitted receipts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        enforcement = _load_json(args.enforcement_evidence) if args.enforcement_evidence else None
        receipt = compose_root_receipt(
            policy=_load_json(args.policy),
            structural_verification=_load_json(args.structural_verification),
            effective_rules_verification=_load_json(args.effective_rules_verification),
            root_subject=_load_json(args.root_subject),
            enforcement_evidence=enforcement,
            root_harness_identity=args.root_harness_identity,
        )
    except RootReceiptError as exc:
        print(f"trusted-main root receipt invalid: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["admission_disposition"] == "AdmittedHistoricalRoot":
        return 0
    return 0 if args.conformance else 3


if __name__ == "__main__":
    raise SystemExit(main())
