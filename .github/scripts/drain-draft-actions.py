#!/usr/bin/env python3
"""Conservatively drain known draft-only GitHub Actions backlog locally.

A runner-backed cancellation workflow cannot repair runner starvation when that
workflow is itself queued. This operator tool uses authenticated `gh api`
requests from a local shell and is dry-run by default.

Safety boundary:
* only explicitly allowlisted workflow paths are considered;
* only queued/requested/waiting/pending runs are considered by default;
* exactly one GitHub-associated PR identity is required for every run;
* that PR must currently be open, draft, and same-repository;
* destructive mode re-reads PR state immediately before every cancellation;
* scientific/qualification workflows are absent from the default allowlist;
* in-progress jobs require a separate explicit flag;
* batches are bounded by an operator cap and live API-rate budget.

Receipts are operational evidence only. They carry no scientific,
qualification, or merge authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlencode

DEFAULT_REPO = "Luminous-Dynamics/symthaea"
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
RECEIPT_SCHEMA = "symthaea.ci-draft-queue-drain.v1"
EVENTS = ("pull_request", "pull_request_target")
DEFAULT_STATUSES = ("queued", "requested", "waiting", "pending")

# Infrastructure workflows whose intended draft semantics are already
# no-runner/maintenance-only. Qualification workflows are deliberately absent.
INFRA_WORKFLOWS = frozenset(
    {
        ".github/workflows/ci.yml",
        ".github/workflows/draft-ci-governor.yml",
        ".github/workflows/pr-governance.yml",
        ".github/workflows/showroom-integrity.yml",
        ".github/workflows/workflow-syntax.yml",
    }
)

# Migration set only. Historical measurement runs can have different semantics,
# so these require explicit --include-specialized.
SPECIALIZED_WORKFLOWS = frozenset(
    {
        ".github/workflows/fractal-time-lab.yml",
        ".github/workflows/broca-measurements.yml",
        ".github/workflows/broca-feature-matrix.yml",
        ".github/workflows/communication-evidence.yml",
        ".github/workflows/coding-backend-regression.yml",
        ".github/workflows/coding-agent-quality.yml",
        ".github/workflows/sym-arch-001.yml",
    }
)

STATUS_PRIORITY = {
    "queued": 0,
    "requested": 1,
    "waiting": 2,
    "pending": 3,
    "in_progress": 4,
}
WORKFLOW_PRIORITY = {
    ".github/workflows/ci.yml": 0,
    ".github/workflows/pr-governance.yml": 1,
    ".github/workflows/workflow-syntax.yml": 1,
    ".github/workflows/showroom-integrity.yml": 1,
    ".github/workflows/draft-ci-governor.yml": 3,
}


class GhError(RuntimeError):
    def __init__(self, message: str, *, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


@dataclass(frozen=True)
class Candidate:
    run_id: int
    workflow_path: str
    workflow_name: str
    event: str
    status: str
    created_at: str
    head_sha: str
    head_branch: str
    pr_number: int


def gh_api(endpoint: str, *, method: str = "GET", expect_json: bool = True) -> Any:
    command = [
        "gh",
        "api",
        "--method",
        method,
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "X-GitHub-Api-Version: 2022-11-28",
        endpoint,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise GhError(
            f"gh api {method} {endpoint} failed with exit {completed.returncode}",
            stderr=completed.stderr.strip(),
        )
    if not expect_json:
        return completed.stdout
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise GhError(f"gh api returned invalid JSON for {endpoint}: {error}") from error


def paged_object_list(
    repo: str,
    resource: str,
    key: str,
    params: dict[str, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for page in range(1, 101):
        query = {**params, "per_page": "100", "page": str(page)}
        endpoint = f"/repos/{repo}/{resource}?{urlencode(query)}"
        payload = gh_api(endpoint)
        if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
            raise GhError(f"{endpoint} did not contain list field {key!r}")
        items = payload[key]
        result.extend(item for item in items if isinstance(item, dict))
        if len(items) < 100:
            return result
    raise GhError(f"refusing to paginate beyond 10,000 items for {resource}")


def paged_array(
    repo: str,
    resource: str,
    params: dict[str, str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for page in range(1, 101):
        query = {**params, "per_page": "100", "page": str(page)}
        endpoint = f"/repos/{repo}/{resource}?{urlencode(query)}"
        payload = gh_api(endpoint)
        if not isinstance(payload, list):
            raise GhError(f"{endpoint} did not return an array")
        result.extend(item for item in payload if isinstance(item, dict))
        if len(payload) < 100:
            return result
    raise GhError(f"refusing to paginate beyond 10,000 items for {resource}")


def open_prs(repo: str) -> dict[int, dict[str, Any]]:
    return {
        int(pr["number"]): pr
        for pr in paged_array(repo, "pulls", {"state": "open"})
        if isinstance(pr.get("number"), int)
    }


def list_live_runs(repo: str, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for event in EVENTS:
        for status in statuses:
            for run in paged_object_list(
                repo,
                "actions/runs",
                "workflow_runs",
                {"event": event, "status": status},
            ):
                run_id = run.get("id")
                if isinstance(run_id, int):
                    by_id[run_id] = run
    return list(by_id.values())


def same_repo_draft(pr: dict[str, Any], repo: str) -> bool:
    head = pr.get("head")
    head_repo = head.get("repo") if isinstance(head, dict) else None
    return (
        pr.get("state") == "open"
        and pr.get("draft") is True
        and isinstance(head_repo, dict)
        and head_repo.get("full_name") == repo
    )


def classify_candidate(
    run: dict[str, Any],
    prs: dict[int, dict[str, Any]],
    repo: str,
    allowlist: frozenset[str],
    statuses: tuple[str, ...],
) -> Candidate | None:
    workflow_path = run.get("path")
    event = run.get("event")
    status = run.get("status")
    if workflow_path not in allowlist or event not in EVENTS or status not in statuses:
        return None

    associations = run.get("pull_requests")
    if not isinstance(associations, list) or len(associations) != 1:
        return None
    association = associations[0]
    if not isinstance(association, dict):
        return None
    pr_number = association.get("number")
    if not isinstance(pr_number, int):
        return None
    pr = prs.get(pr_number)
    if pr is None or not same_repo_draft(pr, repo):
        return None

    run_id = run.get("id")
    if not isinstance(run_id, int):
        return None
    return Candidate(
        run_id=run_id,
        workflow_path=str(workflow_path),
        workflow_name=str(run.get("name") or workflow_path),
        event=str(event),
        status=str(status),
        created_at=str(run.get("created_at") or ""),
        head_sha=str(run.get("head_sha") or ""),
        head_branch=str(run.get("head_branch") or ""),
        pr_number=pr_number,
    )


def fetch_pr(repo: str, number: int) -> dict[str, Any]:
    payload = gh_api(f"/repos/{repo}/pulls/{number}")
    if not isinstance(payload, dict):
        raise GhError(f"pull request #{number} did not return an object")
    return payload


def live_rate_budget() -> tuple[int, int, str]:
    payload = gh_api("/rate_limit")
    resources = payload.get("resources") if isinstance(payload, dict) else None
    core = resources.get("core") if isinstance(resources, dict) else None
    if not isinstance(core, dict):
        raise GhError("rate-limit response missing resources.core")
    remaining = int(core.get("remaining", 0))
    reset = int(core.get("reset", 0))
    reset_iso = datetime.fromtimestamp(reset, tz=timezone.utc).isoformat()
    reserve = 200
    # Each candidate can consume one PR re-read and one cancellation request.
    return remaining, max(0, (remaining - reserve) // 2), reset_iso


def cancel_run(repo: str, run_id: int) -> str:
    try:
        gh_api(
            f"/repos/{repo}/actions/runs/{run_id}/cancel",
            method="POST",
            expect_json=False,
        )
        return "accepted"
    except GhError as error:
        lowered = error.stderr.lower()
        if "http 409" in lowered or "status code 409" in lowered:
            return "already_non_cancellable"
        raise


def candidate_sort_key(candidate: Candidate) -> tuple[int, int, str, int]:
    return (
        STATUS_PRIORITY.get(candidate.status, 99),
        WORKFLOW_PRIORITY.get(candidate.workflow_path, 2),
        candidate.created_at,
        candidate.run_id,
    )


def write_receipt(path: str | None, receipt: dict[str, Any]) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"receipt={destination}")


def self_test() -> None:
    repo = DEFAULT_REPO
    draft_pr = {
        "number": 7,
        "state": "open",
        "draft": True,
        "head": {"repo": {"full_name": repo}},
    }
    run = {
        "id": 99,
        "path": ".github/workflows/ci.yml",
        "name": "CI",
        "event": "pull_request",
        "status": "queued",
        "created_at": "2026-01-01T00:00:00Z",
        "head_sha": "a" * 40,
        "head_branch": "feature",
        "pull_requests": [{"number": 7}],
    }

    candidate = classify_candidate(
        run, {7: draft_pr}, repo, INFRA_WORKFLOWS, DEFAULT_STATUSES
    )
    assert candidate is not None and candidate.pr_number == 7

    ready = dict(draft_pr)
    ready["draft"] = False
    assert classify_candidate(
        run, {7: ready}, repo, INFRA_WORKFLOWS, DEFAULT_STATUSES
    ) is None

    fork = dict(draft_pr)
    fork["head"] = {"repo": {"full_name": "someone/fork"}}
    assert classify_candidate(
        run, {7: fork}, repo, INFRA_WORKFLOWS, DEFAULT_STATUSES
    ) is None

    unknown = dict(run)
    unknown["path"] = ".github/workflows/scientific-qualification.yml"
    assert classify_candidate(
        unknown, {7: draft_pr}, repo, INFRA_WORKFLOWS, DEFAULT_STATUSES
    ) is None

    ambiguous = dict(run)
    ambiguous["pull_requests"] = [{"number": 7}, {"number": 8}]
    assert classify_candidate(
        ambiguous, {7: draft_pr}, repo, INFRA_WORKFLOWS, DEFAULT_STATUSES
    ) is None

    running = dict(run)
    running["status"] = "in_progress"
    assert classify_candidate(
        running, {7: draft_pr}, repo, INFRA_WORKFLOWS, DEFAULT_STATUSES
    ) is None
    assert classify_candidate(
        running,
        {7: draft_pr},
        repo,
        INFRA_WORKFLOWS,
        DEFAULT_STATUSES + ("in_progress",),
    ) is not None

    specialized = dict(run)
    specialized["path"] = ".github/workflows/broca-measurements.yml"
    assert classify_candidate(
        specialized, {7: draft_pr}, repo, INFRA_WORKFLOWS, DEFAULT_STATUSES
    ) is None
    assert classify_candidate(
        specialized,
        {7: draft_pr},
        repo,
        frozenset(set(INFRA_WORKFLOWS) | set(SPECIALIZED_WORKFLOWS)),
        DEFAULT_STATUSES,
    ) is not None

    assert INFRA_WORKFLOWS.isdisjoint(SPECIALIZED_WORKFLOWS)
    print("draft_queue_drain_self_test=PASS")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Dry-run or cancel allowlisted Actions runs associated with open "
            "same-repository draft PRs."
        )
    )
    root.add_argument("--repo", default=DEFAULT_REPO)
    root.add_argument("--apply", action="store_true")
    root.add_argument("--max-cancellations", type=int, default=100)
    root.add_argument("--include-specialized", action="store_true")
    root.add_argument("--include-in-progress", action="store_true")
    root.add_argument("--receipt", help="optional JSON receipt path")
    root.add_argument("--self-test", action="store_true")
    return root


def report_error(error: GhError) -> None:
    print(str(error), file=sys.stderr)
    if error.stderr:
        print(error.stderr, file=sys.stderr)


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0

    if not REPO_RE.fullmatch(args.repo):
        print(f"invalid --repo {args.repo!r}; expected owner/name", file=sys.stderr)
        return 2
    if not 1 <= args.max_cancellations <= 300:
        print("--max-cancellations must be in [1, 300]", file=sys.stderr)
        return 2

    allowlist = INFRA_WORKFLOWS
    if args.include_specialized:
        allowlist = frozenset(set(allowlist) | set(SPECIALIZED_WORKFLOWS))
    statuses = DEFAULT_STATUSES + (("in_progress",) if args.include_in_progress else ())

    try:
        prs = open_prs(args.repo)
        runs = list_live_runs(args.repo, statuses)
    except FileNotFoundError:
        print("gh CLI is required and was not found on PATH", file=sys.stderr)
        return 2
    except GhError as error:
        report_error(error)
        return 2

    candidates = [
        candidate
        for run in runs
        if (candidate := classify_candidate(run, prs, args.repo, allowlist, statuses))
        is not None
    ]
    candidates.sort(key=candidate_sort_key)
    by_workflow = Counter(candidate.workflow_path for candidate in candidates)
    by_status = Counter(candidate.status for candidate in candidates)

    print(f"repo={args.repo}")
    print(f"mode={'apply' if args.apply else 'dry-run'}")
    print(f"open_prs={len(prs)}")
    print(f"live_runs_examined={len(runs)}")
    print(f"eligible_allowlisted_draft_runs={len(candidates)}")
    print(f"include_specialized={str(args.include_specialized).lower()}")
    print(f"include_in_progress={str(args.include_in_progress).lower()}")
    for path, count in sorted(by_workflow.items()):
        print(f"eligible_workflow[{path}]={count}")
    for status, count in sorted(by_status.items()):
        print(f"eligible_status[{status}]={count}")

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "authority": "operator-maintenance-only",
        "scientific_claim": "NONE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": args.repo,
        "mode": "apply" if args.apply else "dry-run",
        "include_specialized": args.include_specialized,
        "include_in_progress": args.include_in_progress,
        "allowlist": sorted(allowlist),
        "statuses": list(statuses),
        "live_runs_examined": len(runs),
        "eligible_runs": len(candidates),
        "eligible_by_workflow": dict(sorted(by_workflow.items())),
        "eligible_by_status": dict(sorted(by_status.items())),
        "requested_max_cancellations": args.max_cancellations,
        "cancellations": [],
    }

    if not args.apply:
        preview = candidates[: args.max_cancellations]
        for candidate in preview:
            print(
                f"[dry-run] run={candidate.run_id} pr={candidate.pr_number} "
                f"status={candidate.status} event={candidate.event} "
                f"workflow={candidate.workflow_path} head={candidate.head_sha}"
            )
        receipt["previewed_runs"] = len(preview)
        write_receipt(args.receipt, receipt)
        return 0

    try:
        remaining, rate_safe_max, reset_iso = live_rate_budget()
    except GhError as error:
        report_error(error)
        return 2

    effective_max = min(args.max_cancellations, rate_safe_max, len(candidates))
    receipt["rate_remaining_before_apply"] = remaining
    receipt["rate_reset_utc"] = reset_iso
    receipt["rate_safe_max"] = rate_safe_max
    receipt["effective_max_cancellations"] = effective_max
    print(
        f"rate_remaining={remaining} rate_safe_max={rate_safe_max} "
        f"effective_max={effective_max} reset={reset_iso}"
    )

    if effective_max == 0:
        print("no safely reserved API budget is available; no cancellations sent")
        write_receipt(args.receipt, receipt)
        return 0

    accepted = 0
    already_gone = 0
    state_changed = 0
    for candidate in candidates[:effective_max]:
        try:
            live_pr = fetch_pr(args.repo, candidate.pr_number)
        except GhError as error:
            print(
                f"[stop] PR re-read failed before run {candidate.run_id}",
                file=sys.stderr,
            )
            report_error(error)
            break

        if not same_repo_draft(live_pr, args.repo):
            state_changed += 1
            print(
                f"[skip] run={candidate.run_id} pr={candidate.pr_number} "
                "is no longer an open same-repository draft"
            )
            receipt["cancellations"].append(
                {
                    "run_id": candidate.run_id,
                    "pr_number": candidate.pr_number,
                    "result": "pr_no_longer_eligible",
                }
            )
            continue

        try:
            result = cancel_run(args.repo, candidate.run_id)
        except GhError as error:
            print(f"[stop] cancellation failed for run {candidate.run_id}", file=sys.stderr)
            report_error(error)
            receipt["cancellations"].append(
                {
                    "run_id": candidate.run_id,
                    "pr_number": candidate.pr_number,
                    "result": "error",
                    "stderr": error.stderr[:500],
                }
            )
            break

        if result == "accepted":
            accepted += 1
        else:
            already_gone += 1
        print(
            f"[{result}] run={candidate.run_id} pr={candidate.pr_number} "
            f"workflow={candidate.workflow_path} status={candidate.status}"
        )
        receipt["cancellations"].append(
            {
                "run_id": candidate.run_id,
                "pr_number": candidate.pr_number,
                "workflow_path": candidate.workflow_path,
                "status_at_discovery": candidate.status,
                "result": result,
            }
        )
        time.sleep(1.0)

    receipt["accepted"] = accepted
    receipt["already_non_cancellable"] = already_gone
    receipt["pr_no_longer_eligible"] = state_changed
    print(f"accepted={accepted}")
    print(f"already_non_cancellable={already_gone}")
    print(f"pr_no_longer_eligible={state_changed}")
    write_receipt(args.receipt, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
