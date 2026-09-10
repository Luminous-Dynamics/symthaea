#!/usr/bin/env python3
"""Strict-time successor for one exact trusted-main P0 ruleset version state.

V1 proves exact reviewed P0 semantics for one GitHub ruleset version while
preserving the provider timestamp as observed text. V2 keeps that theorem and
adds one narrower fact: the provider timestamp is a calendar-valid canonical
UTC instant with an integer-only ordering coordinate.

This still does not prove which version covered a rule-suite attempt, external
chronology, current admission, or any scientific authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import trusted_main_provider_time as provider_time
import trusted_main_ruleset as p0
import trusted_main_ruleset_version_state as v1

SCHEMA = "symthaea.github-trusted-main-ruleset-version-state.v2"
DOMAIN = b"symthaea.github-trusted-main-ruleset-version-state.v2\0"


class RulesetVersionStateV2Error(ValueError):
    """Fail-closed strict-time version-state validation error."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical(value)).hexdigest()


def verify_version_state_v2(raw: Any, policy: Any) -> dict[str, Any]:
    base = v1.verify_version_state(raw, policy)
    try:
        instant = provider_time.parse_github_utc_instant(
            base["provider_updated_at"], where="provider_updated_at"
        )
    except provider_time.ProviderTimeError as exc:
        raise RulesetVersionStateV2Error(str(exc)) from exc

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "policy_id": base["policy_id"],
        "repository": base["repository"],
        "repository_id": base["repository_id"],
        "target_ref": base["target_ref"],
        "ruleset_id": base["ruleset_id"],
        "version_id": base["version_id"],
        "provider_updated_at": base["provider_updated_at"],
        "provider_unix_nanos": instant.unix_nanos,
        "provider_time_schema": instant.schema,
        "provider_actor_id": base["provider_actor_id"],
        "provider_actor_type": base["provider_actor_type"],
        "state": base["state"],
        "v1_version_state_id": base["version_state_id"],
        "disposition": "P0RulesetVersionStateSatisfied",
        "provider_order_authority": "github-provider-valid-utc-instant-only",
        "chronology_authority": "not-externally-anchored",
        "current_admission": "not-evaluated",
        "scientific_authority": "none",
    }
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
        result = verify_version_state_v2(v1._load_json(args.version_state), v1._load_json(args.policy))
    except (v1.RulesetVersionStateError, p0.PolicyError, RulesetVersionStateV2Error) as exc:
        print(f"trusted-main ruleset version state V2 invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
