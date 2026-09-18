#!/usr/bin/env python3
"""Deterministic CI Admission V1 router.

Routing only. This module never establishes QualificationPassed or merge authority.
For pull_request / merge_group events it can derive whether the candidate modifies
CI control-plane files by comparing exact base/head Git commits.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

QUALIFICATION_LABEL = "qualification-requested"
ALLOWED_PR_ACTIONS = frozenset(
    {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "converted_to_draft",
        "labeled",
        "unlabeled",
        "closed",
    }
)
SHA_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
CONTROL_PLANE_PREFIXES = (".github/workflows/",)
CONTROL_PLANE_EXACT = frozenset(
    {
        "docs/qualification-admissions/ci-admission-v1.json",
        "docs/qualification-admissions/ci-admission-v1-job-census.json",
        "scripts/ci-admission-v1.py",
        "scripts/render-ci-admission-v1.py",
        "scripts/verify-ci-admission-v1-census.py",
    }
)


@dataclass(frozen=True)
class Decision:
    disposition: str
    heavy_eligible: bool
    maintenance_eligible: bool
    exact_subject_required: bool
    authority_mode: str = "RoutingOnly"
    merge_authority: bool = False

    def as_dict(
        self,
        *,
        event_name: str,
        action: str,
        head_sha: str,
        base_sha: str,
        policy_sha: str,
        control_plane_modified: bool,
    ) -> dict:
        return {
            "schema": "symthaea.ci-admission.decision.v1",
            "disposition": self.disposition,
            "heavy_eligible": self.heavy_eligible,
            "maintenance_eligible": self.maintenance_eligible,
            "exact_subject_required": self.exact_subject_required,
            "event_name": event_name,
            "action": action,
            "head_sha": head_sha,
            "base_sha": base_sha,
            "policy_sha": policy_sha,
            "control_plane_modified": control_plane_modified,
            "authority_mode": self.authority_mode,
            "merge_authority": self.merge_authority,
            "claim_boundary": "routing_only_not_qualification_pass",
        }


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"", "false", "null", "none", "0"}:
        return False
    if normalized in {"true", "1"}:
        return True
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_labels(raw: str) -> list[str]:
    if not raw.strip():
        return []
    value = json.loads(raw)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("labels must be a JSON array of strings or null")
    return value


def validate_sha(name: str, value: str, *, required: bool) -> None:
    if not value:
        if required:
            raise ValueError(f"{name} is required")
        return
    if not SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 40- or 64-hex Git object id")


def is_control_plane_path(path: str) -> bool:
    return path in CONTROL_PLANE_EXACT or any(
        path.startswith(prefix) for prefix in CONTROL_PLANE_PREFIXES
    )


def detect_control_plane_change(repository: Path, base_sha: str, head_sha: str) -> bool:
    validate_sha("base_sha", base_sha, required=True)
    validate_sha("head_sha", head_sha, required=True)
    for sha in (base_sha, head_sha):
        subprocess.run(
            ["git", "-C", str(repository), "cat-file", "-e", f"{sha}^{{commit}}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            base_sha,
            head_sha,
            "--",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths = [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]
    return any(is_control_plane_path(path) for path in paths)


def decide(
    *,
    event_name: str,
    action: str,
    is_draft: bool,
    labels: Iterable[str],
    ref: str,
    control_plane_modified: bool,
) -> Decision:
    labels = frozenset(labels)

    if event_name == "pull_request":
        if action not in ALLOWED_PR_ACTIONS:
            return Decision("UnsupportedEvent", False, False, False)
        if action == "closed":
            return Decision("QualificationNotRequested", False, False, False)
        if control_plane_modified:
            return Decision("ControlPlaneReviewRequired", False, False, True)
        if is_draft and QUALIFICATION_LABEL not in labels:
            return Decision("DraftNotQualificationEligible", False, False, False)
        return Decision("QualificationPending", True, False, True)

    if event_name == "merge_group":
        if control_plane_modified:
            return Decision("ControlPlaneReviewRequired", False, False, True)
        return Decision("QualificationPending", True, False, True)

    if event_name == "workflow_dispatch":
        return Decision("QualificationPending", True, False, True)

    if event_name == "schedule":
        return Decision("ScheduledMaintenanceOnly", False, True, True)

    if event_name == "push" and ref == "refs/heads/main":
        return Decision("QualificationPending", True, False, True)

    return Decision("QualificationNotRequested", False, False, False)


def emit(
    decision: Decision,
    *,
    event_name: str,
    action: str,
    head_sha: str,
    base_sha: str,
    policy_sha: str,
    control_plane_modified: bool,
    github_output: Path | None,
) -> None:
    validate_sha("head_sha", head_sha, required=decision.exact_subject_required)
    validate_sha("policy_sha", policy_sha, required=decision.exact_subject_required)
    validate_sha("base_sha", base_sha, required=False)

    data = decision.as_dict(
        event_name=event_name,
        action=action,
        head_sha=head_sha,
        base_sha=base_sha,
        policy_sha=policy_sha,
        control_plane_modified=control_plane_modified,
    )
    print(json.dumps(data, sort_keys=True, separators=(",", ":")))
    if github_output is None:
        return

    values = {
        "disposition": data["disposition"],
        "heavy_eligible": str(data["heavy_eligible"]).lower(),
        "maintenance_eligible": str(data["maintenance_eligible"]).lower(),
        "exact_subject_required": str(data["exact_subject_required"]).lower(),
        "head_sha": data["head_sha"],
        "policy_sha": data["policy_sha"],
        "control_plane_modified": str(data["control_plane_modified"]).lower(),
        "authority_mode": data["authority_mode"],
        "merge_authority": str(data["merge_authority"]).lower(),
    }
    with github_output.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"output {key} contains a newline")
            handle.write(f"{key}={value}\n")


def _self_test() -> None:
    cases = [
        ("draft-open", dict(event_name="pull_request", action="opened", is_draft=True, labels=[], ref="", control_plane_modified=False), "DraftNotQualificationEligible", False, False),
        ("draft-sync", dict(event_name="pull_request", action="synchronize", is_draft=True, labels=[], ref="", control_plane_modified=False), "DraftNotQualificationEligible", False, False),
        ("draft-labeled", dict(event_name="pull_request", action="labeled", is_draft=True, labels=[QUALIFICATION_LABEL], ref="", control_plane_modified=False), "QualificationPending", True, False),
        ("ready-open", dict(event_name="pull_request", action="opened", is_draft=False, labels=[], ref="", control_plane_modified=False), "QualificationPending", True, False),
        ("ready-transition", dict(event_name="pull_request", action="ready_for_review", is_draft=False, labels=[], ref="", control_plane_modified=False), "QualificationPending", True, False),
        ("converted-to-draft", dict(event_name="pull_request", action="converted_to_draft", is_draft=True, labels=[], ref="", control_plane_modified=False), "DraftNotQualificationEligible", False, False),
        ("control-plane", dict(event_name="pull_request", action="synchronize", is_draft=False, labels=[], ref="", control_plane_modified=True), "ControlPlaneReviewRequired", False, False),
        ("unsupported-pr-action", dict(event_name="pull_request", action="assigned", is_draft=False, labels=[], ref="", control_plane_modified=False), "UnsupportedEvent", False, False),
        ("closed", dict(event_name="pull_request", action="closed", is_draft=False, labels=[], ref="", control_plane_modified=True), "QualificationNotRequested", False, False),
        ("manual", dict(event_name="workflow_dispatch", action="", is_draft=False, labels=[], ref="", control_plane_modified=False), "QualificationPending", True, False),
        ("merge-group", dict(event_name="merge_group", action="checks_requested", is_draft=False, labels=[], ref="", control_plane_modified=False), "QualificationPending", True, False),
        ("merge-group-control", dict(event_name="merge_group", action="checks_requested", is_draft=False, labels=[], ref="", control_plane_modified=True), "ControlPlaneReviewRequired", False, False),
        ("main-push", dict(event_name="push", action="", is_draft=False, labels=[], ref="refs/heads/main", control_plane_modified=False), "QualificationPending", True, False),
        ("branch-push", dict(event_name="push", action="", is_draft=False, labels=[], ref="refs/heads/feature", control_plane_modified=False), "QualificationNotRequested", False, False),
        ("schedule", dict(event_name="schedule", action="", is_draft=False, labels=[], ref="", control_plane_modified=False), "ScheduledMaintenanceOnly", False, True),
    ]
    for name, kwargs, expected_disposition, expected_heavy, expected_maintenance in cases:
        got = decide(**kwargs)
        assert got.disposition == expected_disposition, (name, got)
        assert got.heavy_eligible is expected_heavy, (name, got)
        assert got.maintenance_eligible is expected_maintenance, (name, got)
        assert got.merge_authority is False, (name, got)
        assert got.disposition != "QualificationPassed", (name, got)

    assert parse_bool("") is False
    assert parse_bool("null") is False
    assert parse_bool("true") is True
    assert parse_labels("") == []
    assert parse_labels("null") == []
    assert parse_labels('["qualification-requested"]') == ["qualification-requested"]

    assert is_control_plane_path(".github/workflows/ci.yml")
    assert is_control_plane_path(".github/workflows/new-heavy.yml")
    assert is_control_plane_path("scripts/ci-admission-v1.py")
    assert not is_control_plane_path("src/lib.rs")
    assert not is_control_plane_path("docs/qualification-admissions/ci-admission-v1-rollout.md")

    exact = "a" * 40
    out = Path("/tmp/ci-admission-v1-self-test-output")
    out.unlink(missing_ok=True)
    decision = decide(
        event_name="schedule",
        action="",
        is_draft=False,
        labels=[],
        ref="",
        control_plane_modified=False,
    )
    emit(
        decision,
        event_name="schedule",
        action="",
        head_sha=exact,
        base_sha="",
        policy_sha=exact,
        control_plane_modified=False,
        github_output=out,
    )
    emitted = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert emitted["heavy_eligible"] == "false"
    assert emitted["maintenance_eligible"] == "true"
    assert emitted["merge_authority"] == "false"

    try:
        emit(
            decision,
            event_name="schedule",
            action="",
            head_sha="not-a-sha",
            base_sha="",
            policy_sha=exact,
            control_plane_modified=False,
            github_output=None,
        )
    except ValueError as exc:
        assert "head_sha" in str(exc)
    else:
        raise AssertionError("exact-subject decisions must reject invalid head SHA")

    print(f"ci_admission_v1_self_test=PASS cases={len(cases)}")
    print("claim_boundary=routing_only_not_github_parser_or_qualification")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-name")
    parser.add_argument("--action", default="")
    parser.add_argument("--draft", default="false")
    parser.add_argument("--labels-json", default="[]")
    parser.add_argument("--ref", default="")
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--policy-sha", default="")
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    if not args.event_name:
        parser.error("--event-name is required unless --self-test is used")

    control_plane_modified = False
    if args.event_name in {"pull_request", "merge_group"} and args.action != "closed":
        if args.repository is None or not args.base_sha:
            raise SystemExit(
                "pull_request/merge_group routing requires --repository and --base-sha "
                "to derive control-plane drift"
            )
        control_plane_modified = detect_control_plane_change(
            args.repository, args.base_sha, args.head_sha
        )

    decision = decide(
        event_name=args.event_name,
        action=args.action,
        is_draft=parse_bool(args.draft),
        labels=parse_labels(args.labels_json),
        ref=args.ref,
        control_plane_modified=control_plane_modified,
    )
    emit(
        decision,
        event_name=args.event_name,
        action=args.action,
        head_sha=args.head_sha,
        base_sha=args.base_sha,
        policy_sha=args.policy_sha,
        control_plane_modified=control_plane_modified,
        github_output=args.github_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
