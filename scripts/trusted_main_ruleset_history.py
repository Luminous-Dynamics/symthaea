#!/usr/bin/env python3
"""Verify a stable, complete provider-history capture for trusted-main P0.

This module is a narrow completeness primitive for #1289. It consumes two
independent pagination passes over the same GitHub repository-ruleset history.
Each pass must start at page 1 and continue through an explicit empty terminal
page. The two normalized histories must match exactly.

The result is content-addressed provider-readback evidence only. It does not
authenticate GitHub, prove that the capture procedure actually contacted the
provider, establish external chronology, or prove which version governed a
rule-suite attempt. Those remain separate authority layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import trusted_main_provider_time as provider_time
import trusted_main_ruleset as p0

CAPTURE_SCHEMA = "symthaea.github-trusted-main-ruleset-history-capture.v1"
SCHEMA = "symthaea.github-trusted-main-ruleset-history.v1"
DOMAIN = b"symthaea.github-trusted-main-ruleset-history.v1\0"
MAX_JSON_BYTES = 4_000_000
CANONICAL_PER_PAGE = 100
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class RulesetHistoryError(ValueError):
    """Fail-closed ruleset-history validation error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RulesetHistoryError(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise RulesetHistoryError(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except RulesetHistoryError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RulesetHistoryError(f"{path}: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical(value)).hexdigest()


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RulesetHistoryError(f"{where}: positive integer required")
    return value


def _string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RulesetHistoryError(f"{where}: canonical non-empty string required")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise RulesetHistoryError(f"{where}: control characters are forbidden")
    return value


def _normalize_history_item(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"version_id", "actor", "updated_at"}:
        raise RulesetHistoryError(f"{where}: closed version_id/actor/updated_at schema required")
    actor = value["actor"]
    if not isinstance(actor, dict) or set(actor) != {"id", "type"}:
        raise RulesetHistoryError(f"{where}.actor: closed id/type schema required")
    version_id = _positive_int(value["version_id"], where=f"{where}.version_id")
    actor_id = _positive_int(actor["id"], where=f"{where}.actor.id")
    actor_type = _string(actor["type"], where=f"{where}.actor.type")
    updated_at = _string(value["updated_at"], where=f"{where}.updated_at")
    try:
        instant = provider_time.parse_github_utc_instant(updated_at, where=f"{where}.updated_at")
    except provider_time.ProviderTimeError as exc:
        raise RulesetHistoryError(str(exc)) from exc
    return {
        "version_id": version_id,
        "provider_updated_at": updated_at,
        "provider_unix_nanos": instant.unix_nanos,
        "provider_actor_id": actor_id,
        "provider_actor_type": actor_type,
    }


def _normalize_pass(value: Any, *, where: str, per_page: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        raise RulesetHistoryError(
            f"{where}: at least one non-empty page plus an explicit empty terminal page required"
        )

    flattened: list[dict[str, Any]] = []
    for index, page in enumerate(value, start=1):
        page_where = f"{where}[{index - 1}]"
        if not isinstance(page, dict) or set(page) != {"page", "items"}:
            raise RulesetHistoryError(f"{page_where}: closed page/items schema required")
        if page["page"] != index:
            raise RulesetHistoryError(f"{page_where}.page: contiguous pagination starting at 1 required")
        items = page["items"]
        if not isinstance(items, list):
            raise RulesetHistoryError(f"{page_where}.items: array required")
        if len(items) > per_page:
            raise RulesetHistoryError(f"{page_where}.items: exceeds declared per_page")
        is_last = index == len(value)
        if is_last and items:
            raise RulesetHistoryError(f"{page_where}: final page must be explicit empty terminal page")
        if not is_last and not items:
            raise RulesetHistoryError(f"{page_where}: empty page before terminal page is ambiguous")
        for item_index, item in enumerate(items):
            flattened.append(
                _normalize_history_item(item, where=f"{page_where}.items[{item_index}]")
            )

    if not flattened:
        raise RulesetHistoryError(f"{where}: ruleset history contains no versions")

    version_ids = [item["version_id"] for item in flattened]
    if len(version_ids) != len(set(version_ids)):
        raise RulesetHistoryError(f"{where}: duplicate version_id in one pagination pass")
    instants = [item["provider_unix_nanos"] for item in flattened]
    if len(instants) != len(set(instants)):
        raise RulesetHistoryError(
            f"{where}: two versions share one provider instant; interval ordering is ambiguous"
        )

    # Do not trust response order as chronology. Canonicalize using the strict
    # provider instant, while retaining version_id as an independent identity.
    return sorted(flattened, key=lambda item: (item["provider_unix_nanos"], item["version_id"]))


def verify_history_capture(capture: Any, policy: Any, *, expected_ruleset_id: int) -> dict[str, Any]:
    p = p0.normalize_policy(policy)
    ruleset_id = _positive_int(expected_ruleset_id, where="expected_ruleset_id")
    if not isinstance(capture, dict) or set(capture) != {
        "schema", "repository", "repository_id", "ruleset_id", "per_page",
        "first_pass", "second_pass", "observation_basis",
    }:
        raise RulesetHistoryError("history capture: closed V1 capture schema required")
    if capture["schema"] != CAPTURE_SCHEMA:
        raise RulesetHistoryError("history capture.schema: unsupported capture schema")
    if capture["repository"] != p["repository"] or capture["repository_id"] != p["repository_id"]:
        raise RulesetHistoryError("history capture: repository identity mismatch")
    if capture["ruleset_id"] != ruleset_id:
        raise RulesetHistoryError("history capture: ruleset ID mismatch")
    if capture["per_page"] != CANONICAL_PER_PAGE:
        raise RulesetHistoryError(f"history capture.per_page: must be {CANONICAL_PER_PAGE}")
    if capture["observation_basis"] != "github-ruleset-history-double-read":
        raise RulesetHistoryError("history capture.observation_basis: unsupported basis")

    first = _normalize_pass(capture["first_pass"], where="first_pass", per_page=CANONICAL_PER_PAGE)
    second = _normalize_pass(capture["second_pass"], where="second_pass", per_page=CANONICAL_PER_PAGE)
    if first != second:
        raise RulesetHistoryError("ruleset history changed or pagination drifted between the two complete passes")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "policy_id": p0.policy_id(p),
        "repository": p["repository"],
        "repository_id": p["repository_id"],
        "target_ref": p["target_ref"],
        "ruleset_id": ruleset_id,
        "per_page": CANONICAL_PER_PAGE,
        "version_count": len(first),
        "versions": first,
        "disposition": "RulesetHistoryStableCompleteReadback",
        "completeness_basis": "two-full-passes-each-ending-in-explicit-empty-page",
        "provider_order_authority": "github-provider-valid-utc-instants-only",
        "capture_authentication": "none",
        "chronology_authority": "not-externally-anchored",
        "current_admission": "not-evaluated",
        "scientific_authority": "none",
    }
    result["history_id"] = _content_id(result)
    return result


def validate_verified_history(value: Any) -> dict[str, Any]:
    """Revalidate a derived history object and its content identity."""
    if not isinstance(value, dict):
        raise RulesetHistoryError("verified history: object required")
    expected_fields = {
        "schema", "policy_id", "repository", "repository_id", "target_ref",
        "ruleset_id", "per_page", "version_count", "versions", "disposition",
        "completeness_basis", "provider_order_authority", "capture_authentication",
        "chronology_authority", "current_admission", "scientific_authority", "history_id",
    }
    if set(value) != expected_fields:
        raise RulesetHistoryError("verified history: closed derived schema required")
    if value["schema"] != SCHEMA or value["disposition"] != "RulesetHistoryStableCompleteReadback":
        raise RulesetHistoryError("verified history: unsupported theorem/disposition")
    if value["per_page"] != CANONICAL_PER_PAGE:
        raise RulesetHistoryError("verified history: non-canonical pagination")
    versions = value["versions"]
    if not isinstance(versions, list) or not versions or value["version_count"] != len(versions):
        raise RulesetHistoryError("verified history: version_count/versions mismatch")
    prior_nanos: int | None = None
    seen_ids: set[int] = set()
    for index, item in enumerate(versions):
        if not isinstance(item, dict) or set(item) != {
            "version_id", "provider_updated_at", "provider_unix_nanos",
            "provider_actor_id", "provider_actor_type",
        }:
            raise RulesetHistoryError(f"verified history.versions[{index}]: closed schema required")
        version_id = _positive_int(item["version_id"], where=f"verified history.versions[{index}].version_id")
        if version_id in seen_ids:
            raise RulesetHistoryError("verified history: duplicate version_id")
        seen_ids.add(version_id)
        try:
            parsed = provider_time.parse_github_utc_instant(
                item["provider_updated_at"], where=f"verified history.versions[{index}].provider_updated_at"
            )
        except provider_time.ProviderTimeError as exc:
            raise RulesetHistoryError(str(exc)) from exc
        if item["provider_unix_nanos"] != parsed.unix_nanos:
            raise RulesetHistoryError("verified history: provider time coordinate mismatch")
        if prior_nanos is not None and parsed.unix_nanos <= prior_nanos:
            raise RulesetHistoryError("verified history: versions must be strictly ordered by provider instant")
        prior_nanos = parsed.unix_nanos
        _positive_int(item["provider_actor_id"], where=f"verified history.versions[{index}].provider_actor_id")
        _string(item["provider_actor_type"], where=f"verified history.versions[{index}].provider_actor_type")
    observed_id = value["history_id"]
    if not isinstance(observed_id, str) or SHA256_RE.fullmatch(observed_id) is None:
        raise RulesetHistoryError("verified history.history_id: sha256 identity required")
    payload = dict(value)
    del payload["history_id"]
    if _content_id(payload) != observed_id:
        raise RulesetHistoryError("verified history: content identity mismatch")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--expected-ruleset-id", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_history_capture(
            _load_json(args.capture), _load_json(args.policy), expected_ruleset_id=args.expected_ruleset_id
        )
    except (RulesetHistoryError, p0.PolicyError) as exc:
        print(f"trusted-main ruleset history invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
