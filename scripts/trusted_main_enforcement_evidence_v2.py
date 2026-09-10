#!/usr/bin/env python3
"""Derive trusted-main behavioral enforcement evidence from GitHub rule suites.

This verifier separates three distinct facts:

* P0 policy/readback says protection is configured and active;
* GitHub rule-suite readbacks show how concrete ref-update attempts were judged;
* current admission/authority remains a separate later theorem.

Detailed rule-suite JSON is hostile input. Positive behavioral corroboration
requires three distinct failed GitHub rule-suite attempts, each bound to the
same exact `main` root and the exact repository-owned P0 ruleset:

* `pull_request`      -> ordinary direct update was rejected;
* `non_fast_forward` -> force-push/non-fast-forward update was rejected;
* `deletion`         -> deletion was rejected.

A generic operator-authored `blocked=true` field is never accepted as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import trusted_main_effective_rules as effective_rules
import trusted_main_ruleset as p0

SCHEMA = "symthaea.github-trusted-main-enforcement-evidence.v2"
DOMAIN = b"symthaea.github-trusted-main-enforcement-evidence.v2\0"
ROOT_SUBJECT_SCHEMA = "symthaea.github-trusted-main-root-subject.v1"
ROOT_SUBJECT_DOMAIN = b"symthaea.github-trusted-main-root-subject.v1\0"
MAX_JSON_BYTES = 2_000_000
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

OPERATIONS = {
    "ordinary_direct_update": "pull_request",
    "force_push": "non_fast_forward",
    "deletion": "deletion",
}

STRUCTURAL_KEYS = frozenset({
    "schema", "policy_id", "repository", "repository_id", "target_ref",
    "ruleset_id", "ruleset_source_type", "ruleset_source", "disposition",
    "violations", "enforcement_claim", "negative_push_test",
    "organization_rules", "scientific_authority", "verification_id",
})
EFFECTIVE_KEYS = frozenset({
    "schema", "policy_id", "repository", "repository_id", "target_ref",
    "ruleset_id", "observed_p0_rule_types", "disposition", "violations",
    "enforcement_claim", "negative_push_test", "scientific_authority",
    "verification_id",
})


class EnforcementEvidenceError(ValueError):
    """Fail-closed rule-suite evidence validation error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise EnforcementEvidenceError(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise EnforcementEvidenceError(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except EnforcementEvidenceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnforcementEvidenceError(f"{path}: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + _canonical(value)).hexdigest()


def _string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnforcementEvidenceError(f"{where}: non-empty string required")
    if value != value.strip():
        raise EnforcementEvidenceError(f"{where}: surrounding whitespace is non-canonical")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise EnforcementEvidenceError(f"{where}: control characters are forbidden")
    return value


def _int(value: Any, *, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EnforcementEvidenceError(f"{where}: integer >= {minimum} required")
    return value


def _sha40(value: Any, *, where: str) -> str:
    value = _string(value, where=where)
    if SHA40_RE.fullmatch(value) is None:
        raise EnforcementEvidenceError(f"{where}: expected 40 lowercase hex")
    return value


def _sha256_id(value: Any, *, where: str) -> str:
    value = _string(value, where=where)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise EnforcementEvidenceError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _closed_object(raw: Any, expected: frozenset[str], *, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EnforcementEvidenceError(f"{where}: object required")
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise EnforcementEvidenceError(
            f"{where}: closed schema required; missing={missing}, unknown={unknown}"
        )
    return dict(raw)


def _root_subject(raw: Any, policy: dict[str, Any]) -> dict[str, Any]:
    expected_keys = frozenset({
        "schema", "repository", "repository_id", "target_ref", "root_sha",
        "root_tree", "observation_basis",
    })
    obj = _closed_object(raw, expected_keys, where="root_subject")
    if obj["schema"] != ROOT_SUBJECT_SCHEMA:
        raise EnforcementEvidenceError(f"root_subject.schema: expected {ROOT_SUBJECT_SCHEMA!r}")
    out = {
        "schema": ROOT_SUBJECT_SCHEMA,
        "repository": _string(obj["repository"], where="root_subject.repository"),
        "repository_id": _int(obj["repository_id"], where="root_subject.repository_id", minimum=1),
        "target_ref": _string(obj["target_ref"], where="root_subject.target_ref"),
        "root_sha": _sha40(obj["root_sha"], where="root_subject.root_sha"),
        "root_tree": _sha40(obj["root_tree"], where="root_subject.root_tree"),
        "observation_basis": _string(obj["observation_basis"], where="root_subject.observation_basis"),
    }
    if out["observation_basis"] != "github-commit-readback":
        raise EnforcementEvidenceError("root_subject.observation_basis: expected github-commit-readback")
    expected_identity = (policy["repository"], policy["repository_id"], policy["target_ref"])
    observed_identity = (out["repository"], out["repository_id"], out["target_ref"])
    if observed_identity != expected_identity:
        raise EnforcementEvidenceError("root_subject: repository/ref identity mismatch")
    return out


def _root_subject_id(root: dict[str, Any]) -> str:
    return _content_id(ROOT_SUBJECT_DOMAIN, root)


def _verification_inputs(
    structural_raw: Any,
    effective_raw: Any,
    *,
    policy: dict[str, Any],
) -> tuple[str, str, int]:
    structural = _closed_object(structural_raw, STRUCTURAL_KEYS, where="structural")
    effective = _closed_object(effective_raw, EFFECTIVE_KEYS, where="effective")
    pid = p0.policy_id(policy)

    observed_structural_id = _sha256_id(
        structural["verification_id"], where="structural.verification_id"
    )
    structural_payload = dict(structural)
    structural_payload.pop("verification_id")
    expected_structural_id = p0._content_id(p0.VERIFY_DOMAIN, structural_payload)
    if observed_structural_id != expected_structural_id:
        raise EnforcementEvidenceError("structural.verification_id: content ID mismatch")

    observed_effective_id = _sha256_id(
        effective["verification_id"], where="effective.verification_id"
    )
    effective_payload = dict(effective)
    effective_payload.pop("verification_id")
    expected_effective_id = effective_rules._verification_id(effective_payload)
    if observed_effective_id != expected_effective_id:
        raise EnforcementEvidenceError("effective.verification_id: content ID mismatch")

    if structural["policy_id"] != pid or effective["policy_id"] != pid:
        raise EnforcementEvidenceError("verification policy ID mismatch")
    if structural["repository"] != policy["repository"] or effective["repository"] != policy["repository"]:
        raise EnforcementEvidenceError("verification repository mismatch")
    if structural["repository_id"] != policy["repository_id"] or effective["repository_id"] != policy["repository_id"]:
        raise EnforcementEvidenceError("verification repository ID mismatch")
    if structural["target_ref"] != policy["target_ref"] or effective["target_ref"] != policy["target_ref"]:
        raise EnforcementEvidenceError("verification target-ref mismatch")
    if structural["disposition"] != "P0StructurallySatisfied" or structural["violations"] != []:
        raise EnforcementEvidenceError("structural protection is not satisfied")
    if effective["disposition"] != "P0EffectiveRulesSatisfied" or effective["violations"] != []:
        raise EnforcementEvidenceError("effective-rule projection is not satisfied")
    if structural["scientific_authority"] != "none" or effective["scientific_authority"] != "none":
        raise EnforcementEvidenceError("verification scientific_authority must remain none")
    ruleset_id = _int(structural["ruleset_id"], where="structural.ruleset_id", minimum=1)
    if effective["ruleset_id"] != ruleset_id:
        raise EnforcementEvidenceError("verification ruleset ID mismatch")
    if structural["ruleset_source_type"] != "Repository" or structural["ruleset_source"] != policy["repository"]:
        raise EnforcementEvidenceError("structural verification is not repository-owned")
    return observed_structural_id, observed_effective_id, ruleset_id


def _suite_observation(
    raw: Any,
    *,
    operation: str,
    expected_rule_type: str,
    policy: dict[str, Any],
    ruleset_id: int,
    root_sha: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if raw is None:
        return None, []
    if not isinstance(raw, dict):
        raise EnforcementEvidenceError(f"{operation} rule suite: object required")

    violations: list[str] = []
    suite_id = _int(raw.get("id"), where=f"{operation}.rule_suite.id", minimum=1)
    actor_id_raw = raw.get("actor_id")
    actor_id: int | None
    if isinstance(actor_id_raw, bool) or not isinstance(actor_id_raw, int) or actor_id_raw < 1:
        violations.append(f"ActorIdentityMissing:{operation}")
        actor_id = None
    else:
        actor_id = actor_id_raw
    actor_name_raw = raw.get("actor_name")
    if not isinstance(actor_name_raw, str) or not actor_name_raw.strip():
        violations.append(f"ActorNameMissing:{operation}")
        actor_name = None
    else:
        actor_name = actor_name_raw.strip()

    # Rule-suite payloads expose the repository name (not owner/name) plus a
    # stable numeric repository ID. Use the numeric ID as the authority anchor;
    # the short name is retained only as a consistency check.
    expected_short_name = policy["repository"].split("/", 1)[-1]
    if raw.get("repository_id") != policy["repository_id"]:
        violations.append(f"RepositoryIdMismatch:{operation}")
    if raw.get("repository_name") != expected_short_name:
        violations.append(f"RepositoryNameMismatch:{operation}")
    if raw.get("ref") != policy["target_ref"]:
        violations.append(f"RefMismatch:{operation}")

    try:
        before_sha = _sha40(raw.get("before_sha"), where=f"{operation}.before_sha")
        after_sha = _sha40(raw.get("after_sha"), where=f"{operation}.after_sha")
    except EnforcementEvidenceError:
        violations.append(f"CommitIdentityMalformed:{operation}")
        before_sha = raw.get("before_sha") if isinstance(raw.get("before_sha"), str) else None
        after_sha = raw.get("after_sha") if isinstance(raw.get("after_sha"), str) else None
    if before_sha != root_sha:
        violations.append(f"RootBeforeShaMismatch:{operation}")

    pushed_at_raw = raw.get("pushed_at")
    if not isinstance(pushed_at_raw, str) or RFC3339_RE.fullmatch(pushed_at_raw) is None:
        violations.append(f"ChronologyMalformed:{operation}")
        pushed_at = None
    else:
        pushed_at = pushed_at_raw

    if raw.get("result") != "fail":
        violations.append(f"RuleSuiteDidNotFail:{operation}")

    evaluations = raw.get("rule_evaluations")
    if not isinstance(evaluations, list):
        raise EnforcementEvidenceError(f"{operation}.rule_evaluations: array required")
    matching: list[dict[str, Any]] = []
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, dict):
            raise EnforcementEvidenceError(f"{operation}.rule_evaluations[{index}]: object required")
        source = evaluation.get("rule_source")
        if not isinstance(source, dict):
            continue
        if (
            source.get("type") == "ruleset"
            and source.get("id") == ruleset_id
            and evaluation.get("rule_type") == expected_rule_type
        ):
            matching.append(evaluation)

    if len(matching) != 1:
        violations.append(f"ExpectedRuleEvaluationCount:{operation}")
        rule_enforcement = None
        rule_result = None
    else:
        match = matching[0]
        rule_enforcement = match.get("enforcement")
        rule_result = match.get("result")
        if rule_enforcement != "active":
            violations.append(f"RuleNotActive:{operation}")
        if rule_result != "fail":
            violations.append(f"RuleDidNotReject:{operation}")

    if violations:
        return None, violations

    return {
        "operation": operation,
        "rule_type": expected_rule_type,
        "rule_suite_id": suite_id,
        "actor_id": actor_id,
        "actor_name": actor_name,
        "before_sha": before_sha,
        "attempted_after_sha": after_sha,
        "pushed_at": pushed_at,
        "suite_result": "fail",
        "rule_enforcement": rule_enforcement,
        "rule_result": rule_result,
        "ruleset_id": ruleset_id,
        "evidence_basis": "github-rule-suite-detailed-readback",
    }, []


def derive_enforcement_evidence(
    *,
    policy: Any,
    structural_verification: Any,
    effective_rules_verification: Any,
    root_subject: Any,
    direct_update_rule_suite: Any | None,
    force_push_rule_suite: Any | None,
    deletion_rule_suite: Any | None,
) -> dict[str, Any]:
    normalized_policy = p0.normalize_policy(policy)
    structural_id, effective_id, ruleset_id = _verification_inputs(
        structural_verification,
        effective_rules_verification,
        policy=normalized_policy,
    )
    root = _root_subject(root_subject, normalized_policy)

    raw_suites = {
        "ordinary_direct_update": direct_update_rule_suite,
        "force_push": force_push_rule_suite,
        "deletion": deletion_rule_suite,
    }
    observations: dict[str, Any] = {}
    violations: list[str] = []
    missing_operations: list[str] = []
    supplied_count = sum(raw is not None for raw in raw_suites.values())

    for operation, rule_type in OPERATIONS.items():
        raw = raw_suites[operation]
        if raw is None:
            observations[operation] = None
            missing_operations.append(operation)
            continue
        observation, operation_violations = _suite_observation(
            raw,
            operation=operation,
            expected_rule_type=rule_type,
            policy=normalized_policy,
            ruleset_id=ruleset_id,
            root_sha=root["root_sha"],
        )
        observations[operation] = observation
        violations.extend(operation_violations)

    verified_suite_ids = [
        observation["rule_suite_id"]
        for observation in observations.values()
        if observation is not None
    ]
    if len(verified_suite_ids) != len(set(verified_suite_ids)):
        violations.append("DuplicateRuleSuiteAcrossOperations")

    verified_count = sum(observation is not None for observation in observations.values())
    if supplied_count == 0:
        disposition = "EnforcementConfiguredOnly"
    elif violations:
        disposition = "EnforcementEvidenceRejected"
    elif verified_count == len(OPERATIONS):
        disposition = "EnforcementBehaviorallyCorroborated"
    else:
        disposition = "EnforcementBehavioralEvidencePartial"

    violations = sorted(set(violations))
    missing_operations = sorted(missing_operations)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": normalized_policy["repository"],
        "repository_id": normalized_policy["repository_id"],
        "target_ref": normalized_policy["target_ref"],
        "ruleset_id": ruleset_id,
        "policy_id": p0.policy_id(normalized_policy),
        "structural_verification_id": structural_id,
        "effective_rules_verification_id": effective_id,
        "root_subject_id": _root_subject_id(root),
        "root_sha": root["root_sha"],
        "operation_observations": observations,
        "missing_operations": missing_operations,
        "disposition": disposition,
        "violations": violations,
        "bypass_assurance": (
            "no-configured-p0-bypass-and-distinct-non-bypass-failures-observed"
            if disposition == "EnforcementBehaviorallyCorroborated"
            and normalized_policy["allowed_bypass_actors"] == []
            else "not-fully-established"
        ),
        "evidence_authority": "server-rule-evaluation-readback-only",
        "chronology_authority": "provider-timestamp-observed-only",
        "current_admission": "not-evaluated",
        "scientific_authority": "none",
    }
    result["evidence_id"] = _content_id(DOMAIN, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("structural_verification", type=Path)
    parser.add_argument("effective_rules_verification", type=Path)
    parser.add_argument("root_subject", type=Path)
    parser.add_argument("--direct-update-rule-suite", type=Path)
    parser.add_argument("--force-push-rule-suite", type=Path)
    parser.add_argument("--deletion-rule-suite", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = derive_enforcement_evidence(
            policy=_load_json(args.policy),
            structural_verification=_load_json(args.structural_verification),
            effective_rules_verification=_load_json(args.effective_rules_verification),
            root_subject=_load_json(args.root_subject),
            direct_update_rule_suite=_load_json(args.direct_update_rule_suite) if args.direct_update_rule_suite else None,
            force_push_rule_suite=_load_json(args.force_push_rule_suite) if args.force_push_rule_suite else None,
            deletion_rule_suite=_load_json(args.deletion_rule_suite) if args.deletion_rule_suite else None,
        )
    except (EnforcementEvidenceError, p0.PolicyError, effective_rules.EffectiveRulesError) as exc:
        print(f"trusted-main enforcement evidence invalid: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["disposition"] == "EnforcementBehaviorallyCorroborated" else 3


if __name__ == "__main__":
    raise SystemExit(main())
