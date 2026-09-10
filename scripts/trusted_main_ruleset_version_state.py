#!/usr/bin/env python3
"""Verify one GitHub repository-ruleset version as exact reviewed P0 state.

This is a narrow provider-history primitive for #1289. It proves only that one
GitHub ruleset `version_id` contains the exact repository-owned P0 semantic
state. It does not establish that the version covered any particular rule-suite
attempt, that GitHub timestamps are externally anchored chronology, or that the
root is currently admissible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import trusted_main_ruleset as p0

SCHEMA = "symthaea.github-trusted-main-ruleset-version-state.v1"
DOMAIN = b"symthaea.github-trusted-main-ruleset-version-state.v1\0"
MAX_JSON_BYTES = 1_000_000
RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class RulesetVersionStateError(ValueError):
    """Fail-closed ruleset-version-state validation error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RulesetVersionStateError(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise RulesetVersionStateError(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except RulesetVersionStateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RulesetVersionStateError(f"{path}: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical(value)).hexdigest()


def _string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RulesetVersionStateError(f"{where}: canonical non-empty string required")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise RulesetVersionStateError(f"{where}: control characters are forbidden")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RulesetVersionStateError(f"{where}: positive integer required")
    return value


def _sorted_unique_strings(value: Any, *, where: str) -> list[str]:
    if not isinstance(value, list):
        raise RulesetVersionStateError(f"{where}: array required")
    items = [_string(item, where=f"{where}[]") for item in value]
    if len(items) != len(set(items)):
        raise RulesetVersionStateError(f"{where}: duplicate values")
    return sorted(items)


def _normalize_bypass(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RulesetVersionStateError("state.bypass_actors: array required")
    items: list[dict[str, Any]] = []
    for index, actor in enumerate(value):
        if not isinstance(actor, dict) or set(actor) != {"actor_id", "actor_type", "bypass_mode"}:
            raise RulesetVersionStateError(f"state.bypass_actors[{index}]: closed actor schema required")
        items.append({
            "actor_id": _positive_int(actor["actor_id"], where=f"state.bypass_actors[{index}].actor_id"),
            "actor_type": _string(actor["actor_type"], where=f"state.bypass_actors[{index}].actor_type"),
            "bypass_mode": _string(actor["bypass_mode"], where=f"state.bypass_actors[{index}].bypass_mode"),
        })
    items.sort(key=lambda item: (item["actor_type"], item["actor_id"], item["bypass_mode"]))
    if len({(i["actor_type"], i["actor_id"], i["bypass_mode"]) for i in items}) != len(items):
        raise RulesetVersionStateError("state.bypass_actors: duplicate actor identity")
    return items


def _normalize_conditions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"ref_name"}:
        raise RulesetVersionStateError("state.conditions: exact ref_name-only schema required for P0")
    ref_name = value["ref_name"]
    if not isinstance(ref_name, dict) or set(ref_name) != {"include", "exclude"}:
        raise RulesetVersionStateError("state.conditions.ref_name: include/exclude required")
    return {
        "ref_name": {
            "include": _sorted_unique_strings(ref_name["include"], where="state.conditions.ref_name.include"),
            "exclude": _sorted_unique_strings(ref_name["exclude"], where="state.conditions.ref_name.exclude"),
        }
    }


def _normalize_rules(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RulesetVersionStateError("state.rules: array required")
    normalized: list[dict[str, Any]] = []
    for index, rule in enumerate(value):
        if not isinstance(rule, dict) or "type" not in rule:
            raise RulesetVersionStateError(f"state.rules[{index}]: typed object required")
        rule_type = _string(rule["type"], where=f"state.rules[{index}].type")
        if rule_type in {"deletion", "non_fast_forward"}:
            if set(rule) != {"type"}:
                raise RulesetVersionStateError(f"state.rules[{index}]: {rule_type} must not carry parameters")
            normalized.append({"type": rule_type})
            continue
        if rule_type != "pull_request" or set(rule) != {"type", "parameters"}:
            raise RulesetVersionStateError(f"state.rules[{index}]: unexpected P0 rule {rule_type!r}")
        params = rule["parameters"]
        expected_keys = {
            "allowed_merge_methods", "dismiss_stale_reviews_on_push",
            "require_code_owner_review", "require_last_push_approval",
            "required_approving_review_count", "required_review_thread_resolution",
        }
        if not isinstance(params, dict) or set(params) != expected_keys:
            raise RulesetVersionStateError("state.pull_request.parameters: exact P0 parameter schema required")
        bool_fields = (
            "dismiss_stale_reviews_on_push", "require_code_owner_review",
            "require_last_push_approval", "required_review_thread_resolution",
        )
        for field in bool_fields:
            if not isinstance(params[field], bool):
                raise RulesetVersionStateError(f"state.pull_request.parameters.{field}: boolean required")
        count = params["required_approving_review_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RulesetVersionStateError("state.pull_request.parameters.required_approving_review_count: non-negative integer required")
        normalized.append({
            "type": "pull_request",
            "parameters": {
                "allowed_merge_methods": _sorted_unique_strings(
                    params["allowed_merge_methods"],
                    where="state.pull_request.parameters.allowed_merge_methods",
                ),
                "dismiss_stale_reviews_on_push": params["dismiss_stale_reviews_on_push"],
                "require_code_owner_review": params["require_code_owner_review"],
                "require_last_push_approval": params["require_last_push_approval"],
                "required_approving_review_count": count,
                "required_review_thread_resolution": params["required_review_thread_resolution"],
            },
        })
    normalized.sort(key=lambda item: item["type"])
    types = [item["type"] for item in normalized]
    if types != ["deletion", "non_fast_forward", "pull_request"]:
        raise RulesetVersionStateError("state.rules: P0 requires exactly deletion, non_fast_forward, pull_request")
    return normalized


def normalize_version_state(raw: Any, policy: Any) -> dict[str, Any]:
    p = p0.normalize_policy(policy)
    if not isinstance(raw, dict) or set(raw) != {"version_id", "actor", "updated_at", "state"}:
        raise RulesetVersionStateError("ruleset version: closed version_id/actor/updated_at/state schema required")

    version_id = _positive_int(raw["version_id"], where="version_id")
    actor = raw["actor"]
    if not isinstance(actor, dict) or set(actor) != {"id", "type"}:
        raise RulesetVersionStateError("actor: closed id/type schema required")
    actor_id = _positive_int(actor["id"], where="actor.id")
    actor_type = _string(actor["type"], where="actor.type")
    updated_at = _string(raw["updated_at"], where="updated_at")
    if RFC3339_RE.fullmatch(updated_at) is None:
        raise RulesetVersionStateError("updated_at: canonical provider RFC3339 UTC required")

    state = raw["state"]
    expected_state_keys = {
        "id", "name", "target", "source_type", "source", "enforcement",
        "bypass_actors", "conditions", "rules",
    }
    if not isinstance(state, dict) or set(state) != expected_state_keys:
        raise RulesetVersionStateError("state: exact P0 semantic state schema required")
    ruleset_id = _positive_int(state["id"], where="state.id")
    normalized_state = {
        "id": ruleset_id,
        "name": _string(state["name"], where="state.name"),
        "target": _string(state["target"], where="state.target"),
        "source_type": _string(state["source_type"], where="state.source_type"),
        "source": _string(state["source"], where="state.source"),
        "enforcement": _string(state["enforcement"], where="state.enforcement"),
        "bypass_actors": _normalize_bypass(state["bypass_actors"]),
        "conditions": _normalize_conditions(state["conditions"]),
        "rules": _normalize_rules(state["rules"]),
    }

    rendered = p0.render_ruleset(p)
    expected_semantics = {
        "name": rendered["name"],
        "target": rendered["target"],
        "enforcement": rendered["enforcement"],
        "bypass_actors": _normalize_bypass(rendered["bypass_actors"]),
        "conditions": _normalize_conditions(rendered["conditions"]),
        "rules": _normalize_rules(rendered["rules"]),
    }
    if normalized_state["source_type"] != "Repository" or normalized_state["source"] != p["repository"]:
        raise RulesetVersionStateError("state: ruleset version is not repository-owned by expected repository")
    for field, expected in expected_semantics.items():
        if normalized_state[field] != expected:
            raise RulesetVersionStateError(f"state.{field}: ruleset version differs from reviewed P0 policy")

    return {
        "schema": SCHEMA,
        "policy_id": p0.policy_id(p),
        "repository": p["repository"],
        "repository_id": p["repository_id"],
        "target_ref": p["target_ref"],
        "ruleset_id": ruleset_id,
        "version_id": version_id,
        "provider_updated_at": updated_at,
        "provider_actor_id": actor_id,
        "provider_actor_type": actor_type,
        "state": normalized_state,
        "disposition": "P0RulesetVersionStateSatisfied",
        "provider_order_authority": "github-provider-observed-only",
        "chronology_authority": "not-externally-anchored",
        "current_admission": "not-evaluated",
        "scientific_authority": "none",
    }


def verify_version_state(raw: Any, policy: Any) -> dict[str, Any]:
    result = normalize_version_state(raw, policy)
    result["version_state_id"] = _content_id(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("version_state", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_version_state(_load_json(args.version_state), _load_json(args.policy))
    except (RulesetVersionStateError, p0.PolicyError) as exc:
        print(f"trusted-main ruleset version state invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
