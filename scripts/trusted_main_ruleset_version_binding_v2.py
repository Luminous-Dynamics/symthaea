#!/usr/bin/env python3
"""Bind exact P0 version interval evidence to verified V2 -> V3 enforcement lineage.

V1 proves that independently selected V3 enforcement observations resolve inside
one exact closed P0 ruleset-version interval. V2 adds the missing ancestry
requirement: the selected V3 evidence must itself be bound to the exact positive
V2 evidence it descends from by an independently selected enforcement-lineage
receipt.

This remains provider-order/readback evidence only. It does not authenticate
GitHub, establish external chronology, prove current admission, or grant
scientific authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import trusted_main_enforcement_evidence_v2 as enforcement_v2
import trusted_main_enforcement_evidence_v3 as enforcement_v3
import trusted_main_enforcement_lineage as enforcement_lineage
import trusted_main_ruleset as p0
import trusted_main_ruleset_version_binding as binding_v1

SCHEMA = "symthaea.github-trusted-main-ruleset-version-binding.v2"
DOMAIN = b"symthaea.github-trusted-main-ruleset-version-binding.v2\0"
MAX_JSON_BYTES = 4_000_000
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CANONICAL_LABELS = {
    "disposition": "P0RulesetVersionBindingWithEnforcementLineageOnly",
    "lineage_basis": "exact-v2-v3-enforcement-lineage-plus-v1-closed-version-interval",
    "provider_order_authority": "github-provider-valid-utc-instants-only",
    "capture_authentication": "none",
    "operator_authentication": "none",
    "receipt_attestation": "none",
    "chronology_authority": "not-externally-anchored",
    "current_admission": "not-evaluated",
    "scientific_authority": "none",
}


class RulesetVersionBindingV2Error(ValueError):
    """Fail-closed lineage-aware version-binding error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RulesetVersionBindingV2Error(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise RulesetVersionBindingV2Error(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except RulesetVersionBindingV2Error:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RulesetVersionBindingV2Error(f"{path}: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical(value)).hexdigest()


def _sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RulesetVersionBindingV2Error(f"{where}: sha256 identity required")
    return value


def _rederive_lineage(
    *,
    policy: Any,
    v2_evidence: Any,
    v3_evidence: Any,
    lineage_receipt: Any,
    expected_v2_evidence_id: str,
    expected_v3_evidence_id: str,
    expected_lineage_id: str,
) -> dict[str, Any]:
    expected_v2 = _sha256(expected_v2_evidence_id, where="expected_v2_evidence_id")
    expected_v3 = _sha256(expected_v3_evidence_id, where="expected_v3_evidence_id")
    expected_lineage = _sha256(expected_lineage_id, where="expected_lineage_id")
    try:
        regenerated = enforcement_lineage.derive_enforcement_lineage(
            policy=policy,
            enforcement_evidence_v2=v2_evidence,
            enforcement_evidence_v3=v3_evidence,
            expected_v2_evidence_id=expected_v2,
            expected_v3_evidence_id=expected_v3,
        )
    except (enforcement_lineage.EnforcementLineageError, p0.PolicyError) as exc:
        raise RulesetVersionBindingV2Error(f"enforcement_lineage: revalidation failed: {exc}") from exc
    if not isinstance(lineage_receipt, dict):
        raise RulesetVersionBindingV2Error("enforcement_lineage: supplied receipt object required")
    if regenerated != lineage_receipt:
        raise RulesetVersionBindingV2Error(
            "enforcement_lineage: supplied bytes differ from exact V2/V3 re-derived lineage"
        )
    if regenerated["lineage_id"] != expected_lineage:
        raise RulesetVersionBindingV2Error(
            "enforcement_lineage: bytes do not match independently selected identity"
        )
    return regenerated


def derive_version_binding_v2(
    *,
    policy: Any,
    history_verification: Any,
    selected_version_state: Any,
    enforcement_evidence_v2: Any,
    enforcement_evidence_v3: Any,
    enforcement_lineage_receipt: Any,
    selected_observations: Any,
    expected_history_id: str,
    expected_version_state_id: str,
    expected_v2_evidence_id: str,
    expected_v3_evidence_id: str,
    expected_lineage_id: str,
) -> dict[str, Any]:
    normalized_policy = p0.normalize_policy(policy)
    expected_v2 = _sha256(expected_v2_evidence_id, where="expected_v2_evidence_id")
    expected_v3 = _sha256(expected_v3_evidence_id, where="expected_v3_evidence_id")
    lineage = _rederive_lineage(
        policy=normalized_policy,
        v2_evidence=enforcement_evidence_v2,
        v3_evidence=enforcement_evidence_v3,
        lineage_receipt=enforcement_lineage_receipt,
        expected_v2_evidence_id=expected_v2,
        expected_v3_evidence_id=expected_v3,
        expected_lineage_id=expected_lineage_id,
    )

    if lineage["v2_evidence_id"] != expected_v2:
        raise RulesetVersionBindingV2Error("enforcement_lineage: exact V2 identity mismatch")
    if lineage["v3_evidence_id"] != expected_v3:
        raise RulesetVersionBindingV2Error("enforcement_lineage: exact V3 identity mismatch")

    try:
        predecessor = binding_v1.derive_version_binding(
            policy=normalized_policy,
            history_verification=history_verification,
            selected_version_state=selected_version_state,
            enforcement_evidence_v3=enforcement_evidence_v3,
            selected_observations=selected_observations,
            expected_history_id=expected_history_id,
            expected_version_state_id=expected_version_state_id,
            expected_enforcement_evidence_id=expected_v3,
        )
    except Exception as exc:  # preserve exact V1 fail-closed semantics without widening authority
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RulesetVersionBindingV2Error(f"binding_v1: revalidation failed: {exc}") from exc

    if predecessor["enforcement_evidence_id"] != lineage["v3_evidence_id"]:
        raise RulesetVersionBindingV2Error("binding_v1 and enforcement lineage select different V3 evidence")
    if predecessor["trusted_enforcement_selection_id"] != lineage["trusted_enforcement_selection_id"]:
        raise RulesetVersionBindingV2Error("binding_v1 and enforcement lineage select different enforcement set")
    if predecessor["selected_rule_suite_observation_ids"] != lineage["selected_rule_suite_observation_ids"]:
        raise RulesetVersionBindingV2Error("binding_v1 and enforcement lineage select different observations")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "policy_id": predecessor["policy_id"],
        "repository": predecessor["repository"],
        "repository_id": predecessor["repository_id"],
        "target_ref": predecessor["target_ref"],
        "ruleset_id": predecessor["ruleset_id"],
        "selected_version_state_id": predecessor["selected_version_state_id"],
        "selected_version_id": predecessor["selected_version_id"],
        "history_id": predecessor["history_id"],
        "v2_evidence_id": lineage["v2_evidence_id"],
        "v3_evidence_id": lineage["v3_evidence_id"],
        "enforcement_lineage_id": lineage["lineage_id"],
        "trusted_enforcement_selection_id": lineage["trusted_enforcement_selection_id"],
        "selected_rule_suite_observation_ids": predecessor["selected_rule_suite_observation_ids"],
        "selected_interval": predecessor["selected_interval"],
        "attempt_resolutions": predecessor["attempt_resolutions"],
        "v1_binding_id": predecessor["binding_id"],
        **CANONICAL_LABELS,
    }
    result["binding_id"] = _content_id(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("history_verification", type=Path)
    parser.add_argument("selected_version_state", type=Path)
    parser.add_argument("enforcement_evidence_v2", type=Path)
    parser.add_argument("enforcement_evidence_v3", type=Path)
    parser.add_argument("enforcement_lineage_receipt", type=Path)
    parser.add_argument("selected_observations", type=Path)
    parser.add_argument("--expected-history-id", required=True)
    parser.add_argument("--expected-version-state-id", required=True)
    parser.add_argument("--expected-v2-evidence-id", required=True)
    parser.add_argument("--expected-v3-evidence-id", required=True)
    parser.add_argument("--expected-lineage-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = derive_version_binding_v2(
            policy=_load_json(args.policy),
            history_verification=_load_json(args.history_verification),
            selected_version_state=_load_json(args.selected_version_state),
            enforcement_evidence_v2=_load_json(args.enforcement_evidence_v2),
            enforcement_evidence_v3=_load_json(args.enforcement_evidence_v3),
            enforcement_lineage_receipt=_load_json(args.enforcement_lineage_receipt),
            selected_observations=_load_json(args.selected_observations),
            expected_history_id=args.expected_history_id,
            expected_version_state_id=args.expected_version_state_id,
            expected_v2_evidence_id=args.expected_v2_evidence_id,
            expected_v3_evidence_id=args.expected_v3_evidence_id,
            expected_lineage_id=args.expected_lineage_id,
        )
    except (RulesetVersionBindingV2Error, p0.PolicyError) as exc:
        print(f"trusted-main ruleset version binding V2 invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
