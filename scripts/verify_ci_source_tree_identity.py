#!/usr/bin/env python3
"""Pure verifier/compiler for CI source-tree identity observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROFILE_SCHEMA = "symthaea-ci-source-tree-identity-profile-v1"
OBS_SCHEMA = "symthaea-ci-source-tree-observation-v1"
IDENTITY_SCHEMA = "symthaea-ci-source-tree-identity-v1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REPO_ID_RE = re.compile(r"^[1-9][0-9]*$")

PROFILE_KEYS = {
    "schema", "pull_request", "repository_identity", "workflow_definition_context",
    "workflow_dispatch", "observation", "separation", "authority",
}
PR_KEYS = {"theorem_qualification", "merge_compatibility"}
LANE_KEYS = {"lane", "checkout_ref_expression", "expected_commit_expression", "meaning"}
REPOSITORY_IDENTITY_KEYS = {
    "base_repository_id_expression", "head_repository_id_expression",
    "retain_base_repository_name", "retain_head_repository_name",
}
WORKFLOW_CONTEXT_KEYS = {
    "workflow_ref_expression", "workflow_sha_expression", "retain_workflow_ref",
    "retain_workflow_sha", "workflow_definition_identity_separate",
}
DISPATCH_KEYS = {"counts_as_exact_pr_head_qualification", "counts_as_pr_merge_compatibility"}
OBS_POLICY_KEYS = {
    "observed_commit_command", "observed_tree_command", "require_observed_commit_equals_expected",
    "event_merge_sha_optional_for_exact_head", "retain_repository_names", "retain_repository_ids",
    "retain_event_head_sha", "retain_event_merge_sha", "retain_observed_tree_sha",
    "tree_sha_is_diagnostic_only",
}
SEPARATION_KEYS = {
    "head_qualification_is_merge_compatibility", "merge_compatibility_is_head_qualification",
    "exact_head_identity_excludes_merge_sha", "exact_head_identity_excludes_tree_sha",
    "exact_head_identity_excludes_workflow_definition", "merge_identity_excludes_head_sha",
    "merge_identity_excludes_tree_sha", "merge_identity_excludes_workflow_definition",
}
AUTHORITY_KEYS = {
    "source_tree_identity_contract_defined", "source_tree_observed", "source_tree_identity_verified",
    "focused_theorem_passed", "merge_compatibility_passed", "scientific_execution_qualified",
    "transform_executed", "fmq010_established", "neural_alignment_established", "consciousness_evidence",
}
OBS_KEYS = {
    "schema", "repository", "repository_id", "head_repository", "head_repository_id",
    "workflow_ref", "workflow_sha", "event_name", "lane", "event_head_sha", "event_merge_sha",
    "observed_checkout_sha", "observed_tree_sha",
}


class SourceIdentityError(ValueError):
    pass


def _closed(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SourceIdentityError(f"{label}: object required")
    if set(value) != keys:
        raise SourceIdentityError(f"{label}: keys mismatch: {sorted(set(value) ^ keys)}")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise SourceIdentityError(f"{label}: exact boolean required")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise SourceIdentityError(f"{label}: 40 lowercase hex required")
    return value


def _repo_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not REPO_ID_RE.fullmatch(value):
        raise SourceIdentityError(f"{label}: canonical positive decimal string required")
    return value


def _repo_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not REPO_RE.fullmatch(value):
        raise SourceIdentityError(f"{label}: owner/name required")
    owner, name = value.split("/", 1)
    if owner in {".", ".."} or name in {".", ".."}:
        raise SourceIdentityError(f"{label}: path-like repository segment forbidden")
    return value


def _workflow_ref(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceIdentityError(f"{label}: non-empty string required")
    if any(ch in value for ch in ("\x00", "\r", "\n")):
        raise SourceIdentityError(f"{label}: control characters forbidden")
    if "/.github/workflows/" not in value or "@" not in value:
        raise SourceIdentityError(f"{label}: GitHub workflow ref required")
    return value


def verify_profile(raw: Any) -> dict[str, Any]:
    p = _closed(raw, PROFILE_KEYS, "profile")
    if p["schema"] != PROFILE_SCHEMA:
        raise SourceIdentityError("profile: schema mismatch")

    pr = _closed(p["pull_request"], PR_KEYS, "profile.pull_request")
    theorem = _closed(pr["theorem_qualification"], LANE_KEYS, "profile.pull_request.theorem_qualification")
    merge = _closed(pr["merge_compatibility"], LANE_KEYS, "profile.pull_request.merge_compatibility")
    expected_theorem = {
        "lane": "exact-pr-head",
        "checkout_ref_expression": "${{ github.event.pull_request.head.sha }}",
        "expected_commit_expression": "${{ github.event.pull_request.head.sha }}",
        "meaning": "theorem-source-identity",
    }
    expected_merge = {
        "lane": "pr-merge-compatibility",
        "checkout_ref_expression": "${{ github.sha }}",
        "expected_commit_expression": "${{ github.sha }}",
        "meaning": "integration-compatibility-only",
    }
    if theorem != expected_theorem:
        raise SourceIdentityError("profile: theorem lane drift")
    if merge != expected_merge:
        raise SourceIdentityError("profile: merge lane drift")

    repository_identity = _closed(
        p["repository_identity"], REPOSITORY_IDENTITY_KEYS, "profile.repository_identity"
    )
    if repository_identity["base_repository_id_expression"] != "${{ github.repository_id }}":
        raise SourceIdentityError("profile: base repository identity expression drift")
    if repository_identity["head_repository_id_expression"] != "${{ github.event.pull_request.head.repo.id }}":
        raise SourceIdentityError("profile: head repository identity expression drift")
    for key in ("retain_base_repository_name", "retain_head_repository_name"):
        if not _bool(repository_identity[key], f"profile.repository_identity.{key}"):
            raise SourceIdentityError(f"profile: {key} must remain true")

    workflow_context = _closed(
        p["workflow_definition_context"], WORKFLOW_CONTEXT_KEYS, "profile.workflow_definition_context"
    )
    if workflow_context["workflow_ref_expression"] != "${{ github.workflow_ref }}":
        raise SourceIdentityError("profile: workflow ref expression drift")
    if workflow_context["workflow_sha_expression"] != "${{ github.workflow_sha }}":
        raise SourceIdentityError("profile: workflow SHA expression drift")
    for key in ("retain_workflow_ref", "retain_workflow_sha", "workflow_definition_identity_separate"):
        if not _bool(workflow_context[key], f"profile.workflow_definition_context.{key}"):
            raise SourceIdentityError(f"profile: {key} must remain true")

    dispatch = _closed(p["workflow_dispatch"], DISPATCH_KEYS, "profile.workflow_dispatch")
    if _bool(dispatch["counts_as_exact_pr_head_qualification"], "dispatch.exact"):
        raise SourceIdentityError("profile: workflow_dispatch cannot count as PR-head qualification")
    if _bool(dispatch["counts_as_pr_merge_compatibility"], "dispatch.merge"):
        raise SourceIdentityError("profile: workflow_dispatch cannot count as PR-merge compatibility")

    obs = _closed(p["observation"], OBS_POLICY_KEYS, "profile.observation")
    if obs["observed_commit_command"] != "git rev-parse HEAD":
        raise SourceIdentityError("profile: observed commit command drift")
    if obs["observed_tree_command"] != "git rev-parse HEAD^{tree}":
        raise SourceIdentityError("profile: observed tree command drift")
    for key in (
        "require_observed_commit_equals_expected", "event_merge_sha_optional_for_exact_head",
        "retain_repository_names", "retain_repository_ids", "retain_event_head_sha",
        "retain_event_merge_sha", "retain_observed_tree_sha", "tree_sha_is_diagnostic_only",
    ):
        if not _bool(obs[key], f"profile.observation.{key}"):
            raise SourceIdentityError(f"profile: {key} must remain true")

    separation = _closed(p["separation"], SEPARATION_KEYS, "profile.separation")
    for key in ("head_qualification_is_merge_compatibility", "merge_compatibility_is_head_qualification"):
        if _bool(separation[key], f"profile.separation.{key}"):
            raise SourceIdentityError("profile: head and merge authority must remain separate")
    for key in (
        "exact_head_identity_excludes_merge_sha", "exact_head_identity_excludes_tree_sha",
        "exact_head_identity_excludes_workflow_definition", "merge_identity_excludes_head_sha",
        "merge_identity_excludes_tree_sha", "merge_identity_excludes_workflow_definition",
    ):
        if not _bool(separation[key], f"profile.separation.{key}"):
            raise SourceIdentityError(f"profile: {key} must remain true")

    authority = _closed(p["authority"], AUTHORITY_KEYS, "profile.authority")
    for key in AUTHORITY_KEYS:
        value = _bool(authority[key], f"profile.authority.{key}")
        if key == "source_tree_identity_contract_defined":
            if not value:
                raise SourceIdentityError("profile: contract-defined authority must be true")
        elif value:
            raise SourceIdentityError(f"profile: premature authority: {key}")

    return p


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compile_observation(profile_raw: Any, observation_raw: Any) -> dict[str, Any]:
    verify_profile(profile_raw)
    o = _closed(observation_raw, OBS_KEYS, "observation")
    if o["schema"] != OBS_SCHEMA:
        raise SourceIdentityError("observation: schema mismatch")

    repository = _repo_name(o["repository"], "observation.repository")
    repository_id = _repo_id(o["repository_id"], "observation.repository_id")
    head_repository = _repo_name(o["head_repository"], "observation.head_repository")
    head_repository_id = _repo_id(o["head_repository_id"], "observation.head_repository_id")
    workflow_ref = _workflow_ref(o["workflow_ref"], "observation.workflow_ref")
    workflow_sha = _sha(o["workflow_sha"], "observation.workflow_sha")

    if o["event_name"] != "pull_request":
        raise SourceIdentityError("observation: only pull_request can establish v1 PR source identity")
    lane = o["lane"]
    if lane not in {"exact-pr-head", "pr-merge-compatibility"}:
        raise SourceIdentityError("observation: unknown lane")

    head_sha = _sha(o["event_head_sha"], "observation.event_head_sha")
    merge_raw = o["event_merge_sha"]
    if merge_raw is None:
        if lane != "exact-pr-head":
            raise SourceIdentityError("observation: merge lane requires event_merge_sha")
        merge_sha = None
    else:
        merge_sha = _sha(merge_raw, "observation.event_merge_sha")

    observed_sha = _sha(o["observed_checkout_sha"], "observation.observed_checkout_sha")
    tree_sha = _sha(o["observed_tree_sha"], "observation.observed_tree_sha")

    expected_sha = head_sha if lane == "exact-pr-head" else merge_sha
    if expected_sha is None:
        raise SourceIdentityError("observation: expected checkout SHA unavailable")
    if observed_sha != expected_sha:
        raise SourceIdentityError(f"observation: checkout commit does not equal expected {lane} commit")

    # The identity root is deliberately lane-minimal. Git commit identity already
    # commits the source tree; the separately retained tree SHA is diagnostic until
    # an independent Git-object reconstruction theorem verifies it. The workflow
    # definition commit/ref are also qualification-envelope context: a separate
    # admission theorem must prove the actually executed workflow bytes equal the
    # registered qualifier blob. Neither context is allowed to rename source.
    source_repository_id = head_repository_id if lane == "exact-pr-head" else repository_id
    identity_preimage = {
        "schema": IDENTITY_SCHEMA,
        "lane": lane,
        "base_repository_id": repository_id,
        "source_repository_id": source_repository_id,
        "checkout_commit_sha": expected_sha,
    }
    digest = hashlib.sha256(canonical_json(identity_preimage)).hexdigest()

    return {
        "schema": IDENTITY_SCHEMA,
        "repository": repository,
        "repository_id": repository_id,
        "head_repository": head_repository,
        "head_repository_id": head_repository_id,
        "workflow_ref": workflow_ref,
        "workflow_sha": workflow_sha,
        "workflow_definition_identity_binding": "separate-qualification-envelope",
        "event_name": "pull_request",
        "lane": lane,
        "event_head_sha": head_sha,
        "event_merge_sha": merge_sha,
        "expected_checkout_sha": expected_sha,
        "observed_checkout_sha": observed_sha,
        "observed_tree_sha": tree_sha,
        "tree_sha_identity_binding": "diagnostic-only",
        "identity_preimage": identity_preimage,
        "identity_sha256": f"sha256:{digest}",
        "authority": {
            "source_tree_identity_verified": True,
            "exact_pr_head_source_verified": lane == "exact-pr-head",
            "pr_merge_source_verified": lane == "pr-merge-compatibility",
            "focused_theorem_passed": False,
            "merge_compatibility_passed": False,
            "scientific_execution_qualified": False,
            "transform_executed": False,
            "fmq010_established": False,
            "neural_alignment_established": False,
            "consciousness_evidence": False,
        },
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SourceIdentityError(f"JSON object: duplicate key: {key}")
        out[key] = value
    return out


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--observation", type=Path)
    args = parser.parse_args(argv)
    try:
        profile = load_json(args.profile)
        result = verify_profile(profile) if args.observation is None else compile_observation(profile, load_json(args.observation))
    except (OSError, json.JSONDecodeError, SourceIdentityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
