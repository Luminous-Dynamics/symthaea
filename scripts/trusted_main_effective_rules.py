#!/usr/bin/env python3
"""Verify GitHub's effective active-rule projection for trusted-main P0.

This is a read-only server-state theorem. It checks that the exact repository-
owned P0 ruleset contributes exactly the required active rules to `main` in
GitHub's `/rules/branches/main` projection. It does not prove behavioral push
rejection, current admission, or any scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import trusted_main_ruleset as p0

SCHEMA = "symthaea.github-trusted-main-effective-rules-verification.v1"
DOMAIN = b"symthaea.github-trusted-main-effective-rules-verification.v1\0"
MAX_JSON_BYTES = 1_000_000


class EffectiveRulesError(ValueError):
    """Fail-closed effective-rules validation error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise EffectiveRulesError(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise EffectiveRulesError(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except EffectiveRulesError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EffectiveRulesError(f"{path}: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _verification_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical(value)).hexdigest()


def verify_effective_rules(policy: Any, ruleset: Any, effective_rules: Any) -> dict[str, Any]:
    p = p0.normalize_policy(policy)
    violations: list[str] = []

    if not isinstance(ruleset, dict):
        raise EffectiveRulesError("ruleset readback: object required")
    ruleset_id = ruleset.get("id")
    if isinstance(ruleset_id, bool) or not isinstance(ruleset_id, int) or ruleset_id < 1:
        raise EffectiveRulesError("ruleset readback: positive integer id required")
    if ruleset.get("name") != p["ruleset_name"]:
        violations.append("RulesetNameMismatch")
    if ruleset.get("source_type") != "Repository":
        violations.append("RulesetSourceTypeMismatch")
    if ruleset.get("source") != p["repository"]:
        violations.append("RulesetSourceMismatch")
    if ruleset.get("enforcement") != "active":
        violations.append("RulesetNotActive")

    if not isinstance(effective_rules, list):
        raise EffectiveRulesError("effective rules: array required")

    matching: list[dict[str, Any]] = []
    for index, rule in enumerate(effective_rules):
        if not isinstance(rule, dict):
            raise EffectiveRulesError(f"effective rules[{index}]: object required")
        rid = rule.get("ruleset_id")
        if rid == ruleset_id:
            matching.append(rule)

    expected_types = set(p["required_rules"])
    observed_types: list[str] = []
    for rule in matching:
        rule_type = rule.get("type")
        if not isinstance(rule_type, str) or not rule_type:
            violations.append("EffectiveRuleTypeMissing")
            continue
        observed_types.append(rule_type)
        if rule.get("ruleset_source_type") != "Repository":
            violations.append(f"EffectiveRuleSourceTypeMismatch:{rule_type}")
        if rule.get("ruleset_source") != p["repository"]:
            violations.append(f"EffectiveRuleSourceMismatch:{rule_type}")

    for expected in sorted(expected_types):
        if observed_types.count(expected) != 1:
            violations.append(f"EffectiveRuleCount:{expected}")

    extras = sorted(set(observed_types) - expected_types)
    for extra in extras:
        violations.append(f"UnexpectedEffectiveRuleFromP0:{extra}")

    violations = sorted(set(violations))
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "policy_id": p0.policy_id(p),
        "repository": p["repository"],
        "repository_id": p["repository_id"],
        "target_ref": p["target_ref"],
        "ruleset_id": ruleset_id,
        "observed_p0_rule_types": sorted(observed_types),
        "disposition": "P0EffectiveRulesSatisfied" if not violations else "P0EffectiveRulesRejected",
        "violations": violations,
        "enforcement_claim": "active-rule-projection-readback-only",
        "negative_push_test": "not-evaluated",
        "scientific_authority": "none",
    }
    result["verification_id"] = _verification_id(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("ruleset", type=Path)
    parser.add_argument("effective_rules", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_effective_rules(
            _load_json(args.policy),
            _load_json(args.ruleset),
            _load_json(args.effective_rules),
        )
    except (EffectiveRulesError, p0.PolicyError) as exc:
        print(f"trusted-main effective rules invalid: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["disposition"] == "P0EffectiveRulesSatisfied" else 3


if __name__ == "__main__":
    raise SystemExit(main())
